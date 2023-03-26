ARG tag="latest"
FROM floopcz/archlinux:${tag}
MAINTAINER Filip Matzner <docker@floop.cz>

RUN pacman -Syu --noconfirm --needed arrayfire ninja cmake boost eigen openmp tbb clang range-v3
RUN ${HOME}/bin/aur-install libcmaes --noconfirm --needed

RUN pacman -U --noconfirm https://archive.archlinux.org/packages/c/cuda/cuda-11.2.2-2-x86_64.pkg.tar.zst \
                          https://archive.archlinux.org/packages/c/cudnn/cudnn-8.1.0.77-1-x86_64.pkg.tar.zst \
                          https://archive.archlinux.org/packages/n/nccl/nccl-2.8.4-1-x86_64.pkg.tar.zst \
                          https://archive.archlinux.org/packages/a/arrayfire/arrayfire-3.8.0-4-x86_64.pkg.tar.zst \
    && echo "IgnorePkg = cuda cudnn nccl arrayfire" >> /etc/pacman.conf \
    && rm -vf /usr/bin/nvidia* \
    && rm -vf /usr/lib/libnvidia* \
    && rm -vf /usr/lib/libcuda* \
    && rm -rvf /opt/cuda/doc/ \
    && rm -rvf /opt/cuda/*nsight* \
    && rm -rvf /opt/cuda/*nvvp* \
    && rm -rvf /opt/cuda/samples/ \
    && source /etc/profile.d/cuda.sh

ADD ./* ${HOME}/echo-state-networks/
WORKDIR echo-state-networks
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build

# start ssh server
# ENTRYPOINT ["/usr/bin/sshd", "-D"]
