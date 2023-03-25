ARG tag="latest"
FROM floopcz/archlinux:${tag}
MAINTAINER Filip Matzner <docker@floop.cz>

RUN pacman -Syu --noconfirm --needed arrayfire ninja cmake boost eigen openmp tbb clang range-v3
RUN ${HOME}/bin/aur-install libcmaes --noconfirm --needed

RUN pacman -U --noconfirm https://archive.archlinux.org/packages/c/cuda/cuda-11.2.2-2-x86_64.pkg.tar.zst \
                          https://archive.archlinux.org/packages/c/cudnn/cudnn-8.1.0.77-1-x86_64.pkg.tar.zst
RUN ln -s /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1


RUN pacman -S --noconfirm --needed graphviz doxygen
RUN mkdir "/tmp/arrayfire" \
    && cd "/tmp/arrayfire" \
    && curl -LO https://raw.githubusercontent.com/archlinux/svntogit-community/packages/arrayfire/trunk/PKGBUILD \
    && curl -LO https://raw.githubusercontent.com/archlinux/svntogit-community/packages/arrayfire/trunk/arrayfire-boost-1.76.0.patch \
    && sed -i 's/\(architecture_build_targets=\).*/\1"6.0;6.1;7.0;7.5;8.0;8.6" \\/' PKGBUILD \
    && chown -Rv nobody "/tmp/arrayfire" \
    && sudo -u nobody XDG_CACHE_HOME="/tmp/.nobody_cache" makepkg \
    && sudo pacman -U --noconfirm "$@" "arrayfire"-*.pkg.tar.* \
    && cd /tmp \
    && sudo rm -rf "/tmp/.nobody_cache" "/tmp/arrayfire"

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
    && source /etc/profile.d/cuda.sh \

ADD ./* ${HOME}/echo-state-networks/
WORKDIR echo-state-networks
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build

# start ssh server
# ENTRYPOINT ["/usr/bin/sshd", "-D"]
