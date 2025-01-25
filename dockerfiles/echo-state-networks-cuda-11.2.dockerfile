# Build as:
# docker builder build --build-arg tag=cuda -t floopcz/echo-state-networks:cuda-11.2 -f dockerfiles/echo-state-networks-cuda-11.2.dockerfile .

# Based on CUDA 11.2.
FROM archlinux:base-devel-20210404.0.18927
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

# Install aur-install
RUN pacman -Syu --noconfirm --needed git
RUN cd /usr/local/bin \
 && curl -LO https://github.com/FloopCZ/linux-environment/raw/master/bin/aur-install \
 && chmod +x ./aur-install

# Install build dependencies
RUN pacman -Syu --noconfirm --needed --asdeps ninja cmake boost eigen tbb clang

# Build fmt
ARG FMT_VERSION="10.2.0-1"
RUN pacman -S --noconfirm --needed --asdeps \
      cmake doxygen git ninja npm python-breathe python-docutils python-jinja python-six python-sphinx python-wheel
RUN cd /tmp/  \
 && sudo -u nobody curl -LO https://gitlab.archlinux.org/archlinux/packaging/packages/fmt/-/archive/${FMT_VERSION}/fmt-${FMT_VERSION}.tar  \
 && sudo -u nobody tar -xvf fmt-${FMT_VERSION}.tar  \
 && cd /tmp/fmt-${FMT_VERSION}  \
 && sed -i "/--target doc/d" PKGBUILD  \
 && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg  \
 && sudo pacman -U --noconfirm --needed fmt-*.pkg.tar.*  \
 && cd /tmp  \
 && sudo rm -rf /tmp/.nobody_cache /tmp/fmt-${FMT_VERSION}.tar /tmp/fmt-${FMT_VERSION}

# Build nlohmann-json
# Dependencies
RUN pacman -S --noconfirm --needed --asdeps cmake git
ARG NLJSON_VERSION="3.11.2-2"
RUN cd /tmp/  \
 && sudo -u nobody curl -LO https://gitlab.archlinux.org/archlinux/packaging/packages/nlohmann-json/-/archive/${NLJSON_VERSION}/nlohmann-json-${NLJSON_VERSION}.tar  \
 && sudo -u nobody tar -xvf nlohmann-json-${NLJSON_VERSION}.tar  \
 && cd /tmp/nlohmann-json-${NLJSON_VERSION}  \
 && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg --skippgpcheck \
 && sudo pacman -U --noconfirm --needed nlohmann-json-*.pkg.tar.*  \
 && cd /tmp  \
 && sudo rm -rf /tmp/.nobody_cache /tmp/nlohmann-json-${NLJSON_VERSION}.tar /tmp/nlohmann-json-${NLJSON_VERSION}

# Build spdlog
ARG SPDLOG_VERSION="1.13.0-1"
RUN cd /tmp/  \
 && sudo -u nobody curl -LO https://gitlab.archlinux.org/archlinux/packaging/packages/spdlog/-/archive/${SPDLOG_VERSION}/spdlog-${SPDLOG_VERSION}.tar  \
 && sudo -u nobody tar -xvf spdlog-${SPDLOG_VERSION}.tar  \
 && cd /tmp/spdlog-${SPDLOG_VERSION}  \
 && sed -i "/--target doc/d" PKGBUILD  \
 && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg  \
 && sudo pacman -U --noconfirm --needed spdlog-*.pkg.tar.*  \
 && cd /tmp  \
 && sudo rm -rf /tmp/.nobody_cache /tmp/spdlog-${SPDLOG_VERSION}.tar /tmp/spdlog-${SPDLOG_VERSION}

# Build arrayfire
ARG AF_VERSION="3.9.0-3"
RUN pacman -S --noconfirm --needed --asdeps cblas fftw lapacke forge freeimage glfw glew intel-mkl \
      cmake graphviz doxygen opencl-headers python ocl-icd cuda cudnn git ninja boost
RUN cd /tmp/  \
 && sudo -u nobody curl -LO https://gitlab.archlinux.org/archlinux/packaging/packages/arrayfire/-/archive/${AF_VERSION}/arrayfire-${AF_VERSION}.tar  \
 && sudo -u nobody tar -xvf arrayfire-${AF_VERSION}.tar  \
 && cd /tmp/arrayfire-${AF_VERSION}  \
 && sed -i "/^options=/d" PKGBUILD  \
 && sed -i "/^depends=/d" PKGBUILD  \
 && sed -i "/^makedepends=/d" PKGBUILD  \
 && sed -i "s/;8.7;8.9;9.0;9.0//" PKGBUILD  \
 && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg  \
 && sudo pacman -U --noconfirm --needed arrayfire-*.pkg.tar.*  \
 && cd /tmp  \
 && sudo rm -rf /tmp/.nobody_cache /tmp/arrayfire-${AF_VERSION}.tar /tmp/arrayfire-${AF_VERSION}

# Install AUR dependencies.
# RUN /root/bin/aur-install clang-git --noconfirm
RUN /usr/local/bin/aur-install range-v3-git --noconfirm --needed
RUN /usr/local/bin/aur-install libcmaes --noconfirm --needed
RUN /usr/local/bin/aur-install googletest-git --noconfirm --needed

# Update glibc
RUN pacman -U --noconfirm --needed \
    https://archive.archlinux.org/packages/g/glibc/glibc-2.35-6-x86_64.pkg.tar.zst

# Compile echo-state-networks.
COPY ./ ${HOME}/echo-state-networks/
WORKDIR ${HOME}/echo-state-networks/
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build
