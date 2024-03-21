# Build as:
# docker builder build --build-arg tag=cuda -t floopcz/echo-state-networks:cuda-11.8 -f echo-state-networks-cuda-11.8.dockerfile .

# Based on CUDA 11.8.
FROM archlinux:base-devel-20221204.0.107760
MAINTAINER Filip Matzner <docker@floop.cz>
RUN echo "Server=https://archive.archlinux.org/repos/2022/12/09/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist
RUN echo "SigLevel = Never" >> /etc/pacman.conf

# Install CUDA.
ARG tag="latest"
RUN if [ "${tag}" = "cuda" ]; then \
    pacman -Syu --noconfirm --needed \
      cuda \
      cudnn \
      nccl \
    && echo "IgnorePkg = cuda cudnn nccl" >> /etc/pacman.conf \
    && rm -vf /usr/bin/nvidia* \
    && rm -vf /usr/lib/libnvidia* \
    && rm -vf /usr/lib/libcuda* \
    && rm -rvf /opt/cuda/doc/ \
    && rm -rvf /opt/cuda/*nsight* \
    && rm -rvf /opt/cuda/*nvvp* \
    && rm -rvf /opt/cuda/samples/ \
    && ln -s /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 \
    && source /etc/profile.d/cuda.sh; \
  fi

# Install dependencis.
RUN pacman -Syu --noconfirm --needed arrayfire fmt ninja cmake boost eigen tbb clang range-v3 gtest

# Install Floop's linux environment.
RUN pacman -Syu --noconfirm --needed git
RUN git clone https://github.com/FloopCZ/linux-environment.git ${HOME}/linux-environment \
  && cd ${HOME}/linux-environment \
  && ./deploy.sh --install

# Install AUR dependencies.
RUN pacman -Syu --noconfirm --needed python-setuptools
RUN /root/bin/aur-install libcmaes --noconfirm --needed

# Compile echo-state-networks.
COPY ./ ${HOME}/echo-state-networks/
WORKDIR ${HOME}/echo-state-networks/
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build
