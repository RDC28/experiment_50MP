#!/usr/bin/env bash
# Best-effort Linux build of Adobe's dng_validate. Usage: build_dng_validate.sh WORKDIR URL SHA256
# The SDK ships Xcode and Visual Studio projects only, so this compiles the sources directly.
# It is unverified until a Colab run; any failure exits nonzero and the caller records a skip.
set -euo pipefail
work="$1"; url="$2"; want="$3"
cd "$work"
[ -f sdk.zip ] || curl -fsSL "$url" -o sdk.zip
got="$(sha256sum sdk.zip | cut -d' ' -f1)"
[ "$got" = "$want" ] || { echo "SDK SHA-256 mismatch: got $got want $want" >&2; exit 2; }
[ -d sdk ] || { mkdir sdk && unzip -q sdk.zip -d sdk; }
root="$(dirname "$(find sdk -type f -name dng_validate.cpp | head -n1)")/.."
root="$(cd "$root" && pwd)"
apt-get install -y -qq g++ libjpeg-dev zlib1g-dev libexpat1-dev >/dev/null
src="$root/dng_sdk/source"
xmp="$(find "$root" -type d -path '*xmp/toolkit' | head -n1)"
[ -n "$xmp" ] || { echo "XMP toolkit not found in SDK" >&2; exit 3; }
mkdir -p obj
flags="-std=c++17 -O2 -w -fPIC -DqLinux=1 -DUNIX_ENV=1 -DXMP_StaticBuild=1 -DqDNGUseLibJPEG=1 -DqDNGSupportJXL=0 -DqDNGThreadSafe=1 -DqDNGValidateTarget=1"
inc="-I$src -I$xmp -I$xmp/public/include -I$xmp/XMPCore/source -I$xmp/XMPFiles/source -I$xmp/source -I$xmp/third-party/zuid/interfaces"
find "$src" -maxdepth 1 -name '*.cpp' ! -name 'dng_validate.cpp' -print0 \
  | xargs -0 -P"$(nproc)" -I{} sh -c 'g++ '"$flags $inc"' -c {} -o obj/$(basename {} .cpp).o'
g++ $flags $inc "$src/dng_validate.cpp" obj/*.o -o dng_validate -ljpeg -lz -lexpat -lpthread
./dng_validate -h >/dev/null 2>&1 || true
test -x dng_validate
