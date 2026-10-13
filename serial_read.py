def leer_esp32(ser):
    """Lee y decodifica la respuesta enviada por la ESP32 (ej. encoders)."""
    if ser is None or not ser.is_open:
        return None

    if ser.in_waiting > 0:
        try:
            respuesta = ser.readline().decode('utf-8').strip()
            if respuesta:
                return respuesta
        except UnicodeDecodeError:
            pass
    return None