FROM python:3.7-alpine3.8

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Build requirements for pip dependencies (also needed for Python hfst module)
RUN apk --no-cache add python3-dev build-base libffi-dev linux-headers \
musl-dev readline-dev

# Build requirements for HFST + Omorfi
RUN apk --no-cache add git autoconf automake libtool file

# Build requirements for HFST
RUN apk --no-cache add flex bison glib glib-dev

# Install HFST
COPY install-hfst.sh /install-hfst.sh
RUN ./install-hfst.sh

# Build requirements for Omorfi
RUN apk --no-cache add zip
RUN pip install hfst

# Install Omorfi
COPY install-omorfi.sh /install-omorfi.sh
RUN ./install-omorfi.sh
COPY local-packages.pth /usr/lib/python3.7/site-packages

# Build requirements for FinnPOS
RUN apk --no-cache add clang clang-dev

# Install FinnPOS
COPY finnpos-makefile.patch /finnpos-makefile.patch
COPY install-finnpos.sh /install-finnpos.sh
RUN ./install-finnpos.sh

# Install pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Clear cache
RUN rm -r /root/.cache
