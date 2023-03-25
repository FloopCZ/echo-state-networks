ARG tag="latest"
FROM floopcz/archlinux:${tag}
MAINTAINER Filip Matzner <docker@floop.cz>

RUN pacman -Syu --noconfirm --needed arrayfire ninja cmake boost eigen openmp tbb clang range-v3
RUN ${HOME}/bin/aur-install libcmaes --noconfirm --needed

RUN ln -s /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

ADD ./* ${HOME}/echo-state-networks/
WORKDIR echo-state-networks
RUN source /etc/profile.d/cuda.sh && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
RUN LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/stubs/ cmake --build build

# start ssh server
# ENTRYPOINT ["/usr/bin/sshd", "-D"]
