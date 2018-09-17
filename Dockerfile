FROM python:3.6-slim

ARG USER_ID
ARG USER_GID
ARG WITH_WAND=''

ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_        US.UTF-8
ENV LANGUAGE        en_US:en
ENV LC_ALL          en_US.UTF-8
ENV PATH            /home/arisu/.local/bin:/usr/local/bin:$PATH

# https://github.com/resin-io-library/base-images/issues/273
RUN mkdir /usr/share/man/man1

RUN apt update \
 && apt install --no-install-recommends -y \
        locales g++ \
        libreoffice-writer libreoffice-core \
        libreoffice-java-common openjdk-8-jre-headless \
        $([ -n "$WITH_WAND" ] && echo 'libmagickwand-dev ghostscript') \
 && rm -rf /var/lib/apt/lists/*


RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen &&\
    locale-gen

# Арису Мизуки - лучший друг Лэйн
RUN groupadd --gid "${USER_GID}" arisu &&\
    useradd --uid "${USER_ID}"  \
            --gid "${USER_GID}" \
            --create-home     \
            --shell /bin/bash \
            arisu

COPY --chown=arisu requirements.txt /app/
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt purge -y g++
USER arisu

COPY --chown=arisu input input/
COPY --chown=arisu main.py .

ENTRYPOINT ["python"]
CMD ["main.py"]
