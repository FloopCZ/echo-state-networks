# Build as:
# docker builder build --build-arg tag=cuda -t floopcz/echo-state-networks:cuda-12.3 -f echo-state-networks-cuda-12.3.dockerfile .

# Based on CUDA 12.3.
FROM archlinux:base-devel-20240101.0.204074
MAINTAINER Filip Matzner <docker@floop.cz>
RUN echo "Server=https://archive.archlinux.org/repos/2024/02/28/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist
RUN echo "SigLevel = Never" >> /etc/pacman.conf

# Update system.
RUN pacman -Syu --noconfirm

# Install CUDA.
ARG tag="latest"
RUN if [ "${tag}" = "cuda" ]; then \
    pacman -S --noconfirm --needed \
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
    && source /etc/profile; \
  fi

# Install dependencis.
RUN pacman -Syu --noconfirm --needed fmt ninja cmake boost eigen tbb clang range-v3 git gtest nlohmann-json

# Install aur-install
RUN cd /usr/local/bin \
 && curl -LO https://github.com/FloopCZ/linux-environment/raw/master/bin/aur-install \
 && chmod +x ./aur-install

# Install arrayfire
RUN cd /tmp \
 && curl -LO http://download.floop.cz/arrayfire-3.9.0-3-x86_64.pkg.tar.zst \
 && pacman -U --noconfirm /tmp/arrayfire-3.9.0-3-x86_64.pkg.tar.zst \
 && rm /tmp/arrayfire-3.9.0-3-x86_64.pkg.tar.zst

# Install AUR dependencies.
RUN /usr/local/bin/aur-install libcmaes-openmp --noconfirm --needed

# Install remote connection utilities
RUN pacman -Syu --noconfirm --needed openssh python python-numpy python-pandas python-pip zsh tmux htop
RUN ssh-keygen -A && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Set up users and passwords.
RUN useradd -m -s /usr/bin/zsh somebody \
  && echo 'somebody ALL=(ALL) ALL' > /etc/sudoers.d/somebody \
  && echo 'Defaults env_keep += "EDITOR"' >> /etc/sudoers.d/somebody \
  && echo "somebody:metaTrailerLeverage" | chpasswd \
  && echo "root:rosenPikeTraversal" | chpasswd

# Compile echo-state-networks.
COPY ./ ${HOME}/echo-state-networks/
WORKDIR ${HOME}/echo-state-networks/
RUN source /etc/profile && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release . \
 && LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build

# Start ssh server by default.
ENTRYPOINT ["/usr/bin/sshd", "-D"]
