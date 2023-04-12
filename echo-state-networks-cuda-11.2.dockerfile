# Build as:
# docker builder build --build-arg tag=cuda -t floopcz/echo-state-networks:cuda-11.2 -f echo-state-networks.dockerfile .
FROM scratch
MAINTAINER Filip Matzner <docker@floop.cz>

# Baed on CUDA 11.2.
COPY --from=archlinux:base-devel-20210404.0.18927 / /
RUN echo "Server=https://archive.archlinux.org/repos/2021/04/05/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist
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
RUN pacman -Syu --noconfirm --needed arrayfire ninja cmake boost eigen openmp tbb clang

# Install Floop's linux environment.
RUN pacman -Syu --noconfirm --needed git
RUN git clone https://github.com/FloopCZ/linux-environment.git ${HOME}/linux-environment \
  && cd ${HOME}/linux-environment \
  && ./deploy.sh --install

# Install AUR dependencies.
# RUN /root/bin/aur-install clang-git --noconfirm
RUN /root/bin/aur-install range-v3-git --noconfirm --needed
RUN /root/bin/aur-install libcmaes --noconfirm --needed

# Compile echo-state-networks.
ADD ./* ${HOME}/echo-state-networks/
WORKDIR ${HOME}/echo-state-networks/
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build
