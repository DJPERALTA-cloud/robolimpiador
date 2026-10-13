def escribir_esp32(ser, m1, m2, m3, m4, pca_states=None):
    """Aplica inversiones físicas de motores y envía el comando de 20 valores a la ESP32."""
    if ser is None or not ser.is_open:
        return

    # Si no se especifican estados para el PCA, se envían -1 para ignorarlos
    if pca_states is None:
        pca_states = [-1] * 16

    # Inversiones físicas para motores locales (M1 y M4 invertidos según tu lógica probada)
    m1_final = int(m1 * -1)
    m2_final = int(m2)
    m3_final = int(m3)
    m4_final = int(m4 * -1)

    # Construimos la trama de 20 valores: 4 Motores + 16 Pines PCA
    trama = [m1_final, m2_final, m3_final, m4_final] + pca_states

    comando = ",".join(map(str, trama)) + "\n"
    ser.write(comando.encode('utf-8'))