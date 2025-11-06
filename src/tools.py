import hashlib
import pandas as pd
import json
import os

# Mismos argumentos que antes
def checksum(dataframe, nombre_clave):
    
    # --- PASO 1: ORDENAR (Indispensable para reproducibilidad) ---
    try:
        df_ordenado = dataframe.astype(str).sort_values(
            by=sorted(dataframe.columns.tolist())
        ).reset_index(drop=True)
    except Exception as e:
        print(f"Error fatal al ordenar el DataFrame: {e}. Abortando.")
        return None

    # --- PASO 2: CONVERTIR A BYTES ---
    try:
        datos_en_bytes = df_ordenado.to_csv(index=False, encoding='utf-8').encode('utf-8')
    except Exception as e:
        print(f"Error al convertir a CSV/bytes: {e}. Abortando.")
        return None

    # --- PASO 3: CALCULAR HASH NUEVO ---
    hash_md5_nuevo = hashlib.md5(datos_en_bytes).hexdigest()
    # LÍNEA CORRECTA (para tu notebook)
    # LÍNEA ÚNICA Y MÁS CONCISA
    ruta_checksum = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'checksums.json')
    
    
    # Preparamos tu 'checksum_info'
    checksum_info_nuevo = {
        'hash': hash_md5_nuevo,
        'nombre_clave': nombre_clave 
    }
    
    # --- PASO 4: LÓGICA "LEER, VERIFICAR HASH, VERIFICAR CLAVE, Y GUARDAR" ---
    datos_json = {} 
    guardar_cambios = False
    # ---> ¡AQUÍ ESTÁ LA CORRECCIÓN! <---
    hash_ya_existe = False # Inicializamos la bandera ANTES del bucle

    # 4a. LEER
    try:
        with open(ruta_checksum, 'r') as f:
            datos_json = json.load(f)
    except FileNotFoundError:
        print(f"No se encontró {ruta_checksum}. Se creará uno nuevo.") # Ajuste mensaje
        datos_json = {} # Aseguramos que sea un dict vacío si no existe
    except json.JSONDecodeError:
        print(f"El archivo {ruta_checksum} está corrupto o vacío. Se sobrescribirá.")
        datos_json = {}

    # 4b. ¡VERIFICAR HASH!
    # Iteramos por todas las claves y valores (info) guardados
    for clave_existente, info_existente in datos_json.items():
        # Asegurarnos que info_existente sea un diccionario y tenga 'hash'
        if isinstance(info_existente, dict) and info_existente.get('hash') == hash_md5_nuevo:
            # ¡Encontramos el mismo HASH!
            hash_ya_existe = True # ¡Ahora sí se asigna aquí si se encuentra!
            if clave_existente == nombre_clave:
                # Es el mismo hash Y la misma clave.
                print(f"Checksum verificado para '{nombre_clave}'. El hash es el mismo. ¡No se guarda nada! 👍")
            else:
                # Es el mismo hash PERO una clave diferente.
                print(f"¡Advertencia! Este mismo hash ya está guardado bajo la clave: '{clave_existente}'.")
                print(f"No se agregará la clave duplicada '{nombre_clave}'.")
            
            guardar_cambios = False # No guardamos
            break # Salimos del bucle

    # 4c. VERIFICAR CLAVE (Solo si el HASH es nuevo)
    # ¡Ahora 'hash_ya_existe' siempre tendrá un valor (True o False)!
    if not hash_ya_existe:
        if nombre_clave in datos_json:
            # El hash es nuevo, pero la clave ya existía (es una actualización)
            # Comparamos el hash viejo por seguridad antes de imprimir cambio
            hash_viejo = datos_json[nombre_clave].get('hash')
            if hash_viejo != hash_md5_nuevo:
                datos_json[nombre_clave] = checksum_info_nuevo
                guardar_cambios = True
            else:
                guardar_cambios = False

        else:
            datos_json[nombre_clave] = checksum_info_nuevo
            guardar_cambios = True

    # 4d. ESCRIBIR (Solo si 'guardar_cambios' es True)
    if guardar_cambios:
        try:
            with open(ruta_checksum, 'w') as f:
                json.dump(datos_json, f, indent=4)
        except Exception as e:
            print(f"Error al guardar el checksum en {ruta_checksum}: {e}")

    # --- CORRECCIÓN FINAL EN EL RETURN ---
    # (Usamos la versión que te di en la respuesta anterior)
    if guardar_cambios:
        return f"Checksum calculado y guardado: {hash_md5_nuevo}"
    else:
        # Añadimos una pequeña verificación por si el hash no se encontró
        mensaje_existencia = f"(ya existía bajo la clave '{clave_existente}', sin cambios)" if hash_ya_existe else "(sin cambios)"
        return f"Checksum verificado: {hash_md5_nuevo} {mensaje_existencia}"