import math
import time

WHEEL_RADIUS = 0.085
TRACK_WIDTH = 0.447

PWMAJUSTE = 1
PWM_NAVEGAR_MIN = 130 * PWMAJUSTE
PWM_NAVEGAR_MAX = 240 * PWMAJUSTE
PWM_BUSCAR = 130 * PWMAJUSTE
PWM_EVASION_RETROCESO = 130 * PWMAJUSTE
PWM_EVASION_ROTACION = 130 * PWMAJUSTE
PWM_DESATASCO = 230 * PWMAJUSTE
PWM_AVANCE_FINAL = 145 * PWMAJUSTE

SMC_K = 18
SMC_EPSILON = 0.005
V_MAX = 1.0
MIN_SPEED_RATIO = 0.5

_CAL_RPM_MAX = 130.0
_CAL_K       = 0.0212
_CAL_PWM0    = 70.20

X_OFFSET_DERECHA_M = 0.05
X_OFFSET_IZQUIERDA_M = 0.05

RPM_MINIMA_MOVIMIENTO = 45.0

EST_BUSCAR = 0
EST_NAVEGAR = 1
EST_DESATASCAR = 2
EST_AVANCE_FINAL = 3
EST_EVITAR_BORDE = 4
EST_ROTAR_BORDE = 5

DT_CONTROL = 0.05
DIST_ACTUADOR_M = 1
PCA_PIN_ACTUADOR = 6
PCA_PWM_ACTUADOR = 4094

DIST_MINIMA_BORDE_M = 0.7
DURACION_RETROCESO_MS = 2000
DURACION_ROTACION_MS = 800

CICLOS_PARA_ATASCO = 50
DURACION_DESATASCO_MS = 800
DURACION_EMPUJE_MS = 400
DURACION_AVANCE_FINAL_MS = 1200

def _pwm_a_rpm(pwm: float) -> float:
    if pwm < _CAL_PWM0: return 0.0
    return _CAL_RPM_MAX * (1.0 - math.exp(-_CAL_K * (pwm - _CAL_PWM0)))

def _rpm_a_pwm(rpm: float, pwm_min: int, pwm_max: int) -> int:
    if rpm <= 1.0: return 0
    rpm = min(rpm, _CAL_RPM_MAX * 0.98)
    pwm = _CAL_PWM0 - math.log(1.0 - rpm / _CAL_RPM_MAX) / _CAL_K
    return int(round(max(pwm_min, min(pwm_max, pwm))))

class NavegacionSMC:
    def __init__(self):
        self.estado = EST_BUSCAR
        self.theta_robot = 0.0
        self._ciclos_atascado = 0
        self._tiempo_desatasco = None
        self._desatasco_derecha = True
        self._tiempo_avance_final = None
        self._tiempo_retroceso = None
        self._tiempo_rotacion = None
        self._dir_rotacion = 1

    def actualizar_odometria(self, rpm_d, rpm_i):
        v_d = rpm_d * (2.0 * math.pi * WHEEL_RADIUS) / 60.0
        v_i = rpm_i * (2.0 * math.pi * WHEEL_RADIUS) / 60.0
        dd = v_d * DT_CONTROL
        di = v_i * DT_CONTROL
        self.theta_robot += (dd - di) / TRACK_WIDTH
        while self.theta_robot > math.pi: self.theta_robot -= 2.0 * math.pi
        while self.theta_robot < -math.pi: self.theta_robot += 2.0 * math.pi

    def calcular_tangente(self, xB, yB):
        return math.atan2(xB, yB)

    def control_smc_proporcional(self, theta_des):
        e = theta_des
        while e > math.pi:  e -= 2.0 * math.pi
        while e < -math.pi: e += 2.0 * math.pi
        steering_factor = math.tanh(SMC_K * e / SMC_EPSILON)
        abs_steering = abs(steering_factor)
        speed_reduction = abs_steering * (1.0 - MIN_SPEED_RATIO)
        if steering_factor > 0:
            v_izq, v_der = V_MAX, V_MAX * (1.0 - speed_reduction)
        else:
            v_der, v_izq = V_MAX, V_MAX * (1.0 - speed_reduction)
        return v_der / WHEEL_RADIUS, v_izq / WHEEL_RADIUS

    def rads_to_pwm(self, rads: float, pwm_min: int, pwm_max: int) -> int:
        rpm = abs(rads) * 60.0 / (2.0 * math.pi)
        return _rpm_a_pwm(rpm, pwm_min, pwm_max) if rpm >= 1.0 else 0

    def _pwm_desatasco(self):
        if self._desatasco_derecha:
            return  PWM_DESATASCO,  PWM_DESATASCO, -PWM_DESATASCO, -PWM_DESATASCO
        else:
            return -PWM_DESATASCO, -PWM_DESATASCO,  PWM_DESATASCO,  PWM_DESATASCO

    def _nombre_estado(self, est):
        estados = {0: "BUSCAR", 1: "NAVEGAR", 2: "DESATASCAR",
                   3: "AVANCE_FINAL", 4: "EVITAR_BORDE", 5: "ROTAR_BORDE"}
        return estados.get(est, "DESCONOCIDO")

    def calcular_control(self, lata_detectada, obj_x, obj_z,
                         rpm_td, rpm_dd, rpm_di, rpm_ti,
                         d_borde_izq, d_borde_cen, d_borde_der):
        
        # Guardar estado antes de evaluar la lógica para detectar cambios
        estado_antes = self.estado
        
        # Ejecutar la lógica de control original
        m1, m2, m3, m4 = self._calcular_control_interno(
            lata_detectada, obj_x, obj_z, rpm_td, rpm_dd, rpm_di, rpm_ti,
            d_borde_izq, d_borde_cen, d_borde_der
        )

        # Imprimir alertas detalladas solo si hubo un cambio de estado
        if self.estado != estado_antes:
            print(f"\n[ALERTA] >>> CAMBIO DE ESTADO: {self._nombre_estado(estado_antes)} -> {self._nombre_estado(self.estado)}")
            if self.estado == EST_DESATASCAR:
                print(f"[DETALLE] Iniciando rutina de desatasco. Dirección hacia la derecha: {self._desatasco_derecha}")
            elif self.estado == EST_EVITAR_BORDE:
                print(f"[DETALLE] Borde detectado cerca. Iniciando retroceso de evasión.")
            elif self.estado == EST_ROTAR_BORDE:
                print(f"[DETALLE] Retroceso finalizado. Iniciando rotación (Dirección: {self._dir_rotacion}).")
            elif self.estado == EST_AVANCE_FINAL:
                print(f"[DETALLE] Meta próxima o lata perdida de vista. Iniciando avance final ciego.")
            print("-" * 50)

        return m1, m2, m3, m4

    def _calcular_control_interno(self, lata_detectada, obj_x, obj_z,
                                  rpm_td, rpm_dd, rpm_di, rpm_ti,
                                  d_borde_izq, d_borde_cen, d_borde_der):

        rpm_d = (abs(rpm_td) + abs(rpm_dd)) / 2.0
        rpm_i = (abs(rpm_ti) + abs(rpm_di)) / 2.0
        self.actualizar_odometria(rpm_d, rpm_i)

        # 1. EVASIÓN DE BORDES (Prioridad máxima)
        if self.estado == EST_EVITAR_BORDE:
            if time.time() - self._tiempo_retroceso < (DURACION_RETROCESO_MS / 1000.0):
                return -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO
            else:
                self.estado = EST_ROTAR_BORDE
                self._tiempo_rotacion = time.time()
                return 0, 0, 0, 0

        if self.estado == EST_ROTAR_BORDE:
            if time.time() - self._tiempo_rotacion < (DURACION_ROTACION_MS / 1000.0):
                pwm_i_rot = PWM_EVASION_ROTACION * self._dir_rotacion
                pwm_d_rot = PWM_EVASION_ROTACION * -self._dir_rotacion
                return pwm_i_rot, pwm_i_rot, pwm_d_rot, pwm_d_rot
            else:
                self.estado = EST_BUSCAR
                return 0, 0, 0, 0

        if (d_borde_izq < DIST_MINIMA_BORDE_M or
            d_borde_cen < DIST_MINIMA_BORDE_M or
            d_borde_der < DIST_MINIMA_BORDE_M):
            self.estado = EST_EVITAR_BORDE
            self._tiempo_retroceso = time.time()
            self._dir_rotacion = -1 if d_borde_izq > d_borde_der else 1
            return -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO, -PWM_EVASION_RETROCESO

        # 2. DESATASCO
        if self.estado == EST_DESATASCAR:
            t = time.time() - self._tiempo_desatasco
            if t < (DURACION_DESATASCO_MS / 1000.0):
                return self._pwm_desatasco()
            elif t < (DURACION_DESATASCO_MS + DURACION_EMPUJE_MS) / 1000.0:
                return PWM_NAVEGAR_MAX, PWM_NAVEGAR_MAX, PWM_NAVEGAR_MAX, PWM_NAVEGAR_MAX
            else:
                self.estado = EST_BUSCAR
                self._ciclos_atascado = 0
                return 0, 0, 0, 0

        # 3. AVANCE FINAL
        if self.estado == EST_AVANCE_FINAL:
            if time.time() - self._tiempo_avance_final < (DURACION_AVANCE_FINAL_MS / 1000.0):
                return PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL
            else:
                self.estado = EST_BUSCAR
                return 0, 0, 0, 0

        # 4. DETECCIÓN DE ATASCO
        encoders = [abs(rpm_td), abs(rpm_dd), abs(rpm_di), abs(rpm_ti)]
        if self.estado == EST_NAVEGAR and sum(1 for r in encoders if r < RPM_MINIMA_MOVIMIENTO) >= 2:
            self._ciclos_atascado += 1
        else:
            self._ciclos_atascado = 0

        if self._ciclos_atascado >= CICLOS_PARA_ATASCO:
            self.estado = EST_DESATASCAR
            self._tiempo_desatasco = time.time()
            self._ciclos_atascado = 0
            self._desatasco_derecha = not self._desatasco_derecha
            return self._pwm_desatasco()

        # 5. SIN LATA: avanzar buscando
        if not lata_detectada or obj_x is None or obj_z is None:
            if self.estado == EST_NAVEGAR:
                self.estado = EST_AVANCE_FINAL
                self._tiempo_avance_final = time.time()
                return PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL
            self.estado = EST_BUSCAR
            return PWM_BUSCAR, PWM_BUSCAR, PWM_BUSCAR, PWM_BUSCAR

        # 6. META ALCANZADA
        dist = math.hypot(obj_x, obj_z)
        if dist <= 0.10:
            self.estado = EST_AVANCE_FINAL
            self._tiempo_avance_final = time.time()
            return PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL, PWM_AVANCE_FINAL

        # 7. NAVEGACIÓN SMC
        self.estado = EST_NAVEGAR
        offset = X_OFFSET_DERECHA_M if obj_x > 0 else -X_OFFSET_IZQUIERDA_M
        theta_des = self.calcular_tangente(obj_x - offset, obj_z)
        w_der, w_izq = self.control_smc_proporcional(theta_des)
        pwm_d = self.rads_to_pwm(w_der, PWM_NAVEGAR_MIN, PWM_NAVEGAR_MAX)
        pwm_i = self.rads_to_pwm(w_izq, PWM_NAVEGAR_MIN, PWM_NAVEGAR_MAX)
        return pwm_i, pwm_i, pwm_d, pwm_d

    def calcular_pca(self):
        """El actuador siempre encendido."""
        pca = [-1] * 16
        pca[PCA_PIN_ACTUADOR] = PCA_PWM_ACTUADOR
        return pca