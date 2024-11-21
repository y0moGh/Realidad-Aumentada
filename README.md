# Realidad Aumentada - Personajes de Marvel

Este proyecto utiliza tecnologías de realidad aumentada para visualizar personajes de Marvel. Aquí aprenderás a configurarlo y ejecutarlo, incluso si no tienes experiencia previa en programación.

---

## Guía Paso a Paso

### 1. Instalar Python

**¿Qué es Python?**  
Es un programa que usaremos para ejecutar este proyecto. Sigue estos pasos para instalarlo:

- Ve a la página oficial de Python: [python.org/downloads]([https://www.python.org/downloads/](https://www.python.org/downloads/release/python-3100/)).  
- Descarga la versión para tu sistema operativo (**Python 3.7 - 3.10**).
- Durante la instalación, asegúrate de marcar la casilla **"Add Python to PATH"**.

Para verificar que se instaló correctamente:
1. Abre tu terminal (o símbolo del sistema en Windows).
2. Escribe:
   ```bash
   python --version
   ```
   Deberías ver un número como **Python 3.x.x**.

---

### 2. Descargar el Proyecto

Necesitamos copiar este proyecto en tu computadora.

1. **Descarga desde GitHub**:
   - Haz clic en este enlace: [Repositorio del Proyecto](https://github.com/y0moGh/Realidad-Aumentada).
   - Presiona el botón verde **"Code"** y selecciona **"Download ZIP"**.
   - Extrae el contenido del archivo ZIP en una carpeta de tu preferencia.

   *Alternativa para usuarios avanzados: Clonar con Git:*
   ```bash
   git clone https://github.com/y0moGh/Realidad-Aumentada.git
   cd Realidad-Aumentada
   ```

---

### 3. Instalar Dependencias

Este proyecto necesita ciertas herramientas para funcionar. Vamos a instalarlas automáticamente usando el archivo `requirements.txt`.

1. Abre tu terminal.
2. Navega a la carpeta donde extrajiste el proyecto:
   ```bash
   cd ruta/donde/esta/el/proyecto
   ```
3. Instala las dependencias con el comando:
   ```bash
   pip install -r requirements.txt
   ```

Esto instalará automáticamente todas las bibliotecas necesarias para que el proyecto funcione.

---

### 4. Ejecutar el Proyecto

1. En la terminal, asegúrate de estar dentro de la carpeta del proyecto:
   ```bash
   cd ruta/donde/esta/el/proyecto
   ```
2. Ejecuta el archivo principal con este comando:
   ```bash
   python main.py
   ```

---

### 5. Ver los Resultados

- Asegúrate de tener una cámara conectada o activa en tu computadora.
- Interactúa con los personajes en la pantalla.

---

## Solución de Problemas

### 1. **Error de "No module named..."**
   Esto significa que falta instalar una biblioteca. Repite el paso 3 para instalarla.

### 2. **La cámara no funciona**
   Verifica que esté conectada y que el programa tenga permisos para usarla.

### 3. **Otros errores**
   Consulta la salida del programa o toma una captura para buscar ayuda. Puedes abrir un **Issue** en el repositorio [aquí](https://github.com/y0moGh/Realidad-Aumentada/issues).

---

## Preguntas Frecuentes (FAQ)

1. **¿Qué es este proyecto?**
   Es una aplicación de realidad aumentada que muestra personajes de Marvel usando tu cámara.

2. **¿Necesito experiencia previa en programación?**
   No, simplemente sigue estos pasos.

3. **¿Puedo contribuir?**
   Sí, envía tus sugerencias o mejoras al repositorio.
