# Etapa de compilación 1: Se utilizó la imagen base ligera de
# Python 3.11.9-slim-bullseye para instalar las dependencias.
FROM python:3.11.9-slim-bullseye AS requirements

# Se estableció el directorio de trabajo en /usr/src/app
# dentro del contenedor.
WORKDIR /usr/src/app

# Se configuró el entorno para evitar la generación de archivos .pyc.
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Se configuró el entorno para obtener un output sin buffer en la terminal.
ENV PYTHONUNBUFFERED=TRUE

# Se actualizó la lista de paquetes, se instaló curl, y se descargó manualmente pip.
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get clean && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    python -m venv /home/venv

# Se añadió el entorno virtual al PATH para que las herramientas instaladas
# sean accesibles.
ENV PATH="/home/venv/bin:$PATH"

# Se copió el archivo requirements.txt, que contiene las dependencias de Python.
COPY requirements.txt .

# Se instalaron las dependencias especificadas en requirements.txt sin
# almacenar en cache.
RUN pip install --no-cache-dir -r requirements.txt

# Etapa de compilación 2: Se utilizó otra imagen base ligera de
# Python para la ejecución de la aplicación.
FROM python:3.11.9-slim-bullseye AS runtime

# Se estableció el directorio de trabajo en /usr/src/app
# dentro del contenedor.
WORKDIR /usr/src/app

# Se configuró el entorno para obtener un output sin buffer en la terminal.
ENV PYTHONUNBUFFERED=TRUE

# Se copió el entorno virtual creado en la primera etapa a esta segunda etapa.
COPY --from=requirements /home/venv /home/venv

# Se añadió nuevamente el entorno virtual al PATH para que las herramientas
# instaladas sean accesibles.
ENV PATH="/home/venv/bin:$PATH"

# Se copió todo el código de la aplicación al directorio de trabajo en
# el contenedor.
COPY . .

# Esta línea expone el puerto 80, permitiendo el acceso HTTP a la aplicación
# desde el exterior.
EXPOSE 80
