ARG tag="latest"
FROM floopcz/archlinux:${tag}
MAINTAINER Filip Matzner <docker@floop.cz>

RUN pacman -Syu --noconfirm --needed arrayfire ninja cmake boost eigen openmp tbb clang range-v3
RUN ${HOME}/bin/aur-install libcmaes --noconfirm --needed

RUN pacman -U --noconfirm https://archive.archlinux.org/packages/c/cuda/cuda-11.8.0-1-x86_64.pkg.tar.zst \
                          https://archive.archlinux.org/packages/c/cudnn/cudnn-8.5.0.96-1-x86_64.pkg.tar.zst
RUN echo "IgnorePkg = cuda cudnn" >> /etc/pacman.conf

RUN pacman -Syu --noconfirm --needed gcc11 graphviz doxygen
RUN mkdir "/tmp/arrayfire" \
    && cd "/tmp/arrayfire" \
    && curl -LO https://raw.githubusercontent.com/archlinux/svntogit-community/packages/arrayfire/trunk/PKGBUILD \
    && curl -LO https://raw.githubusercontent.com/archlinux/svntogit-community/packages/arrayfire/trunk/arrayfire-boost-1.76.0.patch \
    && sed -i 's/\(architecture_build_targets=\).*/\1"6.0;6.1;7.0;7.5;8.0;8.6" \\/' PKGBUILD \
    && sed -i 's/ctest/# ctest/' PKGBUILD \
    && chown -Rv nobody "/tmp/arrayfire" \
    && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg \
    && sudo pacman -U --noconfirm "$@" "arrayfire"-*.pkg.tar.* \
    && cd /tmp \
    && sudo rm -rf "/tmp/.nobody_cache" "/tmp/arrayfire"
RUN echo "IgnorePkg = arrayfire" >> /etc/pacman.conf

RUN pacman --noconfirm --needed -Syu \
      cuda \
      cudnn \
      nccl \
    && rm -vf /usr/bin/nvidia* \
    && rm -vf /usr/lib/libnvidia* \
    && rm -vf /usr/lib/libcuda* \
    && rm -rvf /opt/cuda/doc/ \
    && rm -rvf /opt/cuda/*nsight* \
    && rm -rvf /opt/cuda/*nvvp* \
    && rm -rvf /opt/cuda/samples/ \
    && ln -s /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 \
    && source /etc/profile.d/cuda.sh

ADD ./* ${HOME}/echo-state-networks/
WORKDIR echo-state-networks
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build

# start ssh server
# ENTRYPOINT ["/usr/bin/sshd", "-D"]
