# Build as:
# docker builder build --build-arg tag=cuda -t floopcz/echo-state-networks:cuda-12.4 -f echo-state-networks-cuda-12.4.dockerfile .

# Based on CUDA 12.4.
FROM archlinux:base-devel-20240101.0.204074
MAINTAINER Filip Matzner <docker@floop.cz>
RUN echo "Server=https://archive.archlinux.org/repos/2024/04/14/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist
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
RUN pacman -Syu --noconfirm --needed fmt ninja cmake boost eigen tbb clang range-v3 git gtest nlohmann-json

# Install aur-install
RUN cd /usr/local/bin \
 && curl -LO https://github.com/FloopCZ/linux-environment/raw/master/bin/aur-install \
 && chmod +x ./aur-install

# Build arrayfire
RUN pacman -S --noconfirm --needed arrayfire

# Install AUR dependencies.
RUN /usr/local/bin/aur-install libcmaes-openmp --noconfirm --needed

# Compile echo-state-networks.
COPY ./ ${HOME}/echo-state-networks/
WORKDIR ${HOME}/echo-state-networks/
RUN source /etc/profile && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build
