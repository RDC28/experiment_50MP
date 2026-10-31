#!/usr/bin/env bash
# Linux build of Adobe's dng_validate. Usage: build_dng_validate.sh WORKDIR URL SHA256
# The SDK ships Xcode and Visual Studio projects only, so this compiles the DNG SDK sources and the
# bundled XMPCore toolkit (XMPCore, XMPCommon, their public glue, shared sources and zuid MD5, as in
# the toolkit's own XMPCore CMake list, minus the Adobe-private XMPDocOps sources) directly, against
# system expat, zlib and libjpeg. DNG 1.7.1 always includes JPEG XL, so the SDK's bundled libjxl is
# built with CMake first. Any failure exits nonzero and the caller records a skip.
set -euo pipefail
work="$1"; url="$2"; want="$3"
cd "$work"
[ -f sdk.zip ] || curl -fsSL "$url" -o sdk.zip
got="$(sha256sum sdk.zip | cut -d' ' -f1)"
[ "$got" = "$want" ] || { echo "SDK SHA-256 mismatch: got $got want $want" >&2; exit 2; }
[ -d sdk ] || { mkdir sdk && unzip -q sdk.zip -d sdk; }
validate="$(find "$PWD/sdk" -type f -name dng_validate.cpp | head -n1)"
[ -n "$validate" ] || { echo "dng_validate.cpp not found in SDK" >&2; exit 3; }
src="$(dirname "$validate")"
root="$(cd "$src/../.." && pwd)"
xmp="$root/xmp/toolkit"
[ -d "$xmp/XMPCore/source" ] || { echo "XMP toolkit not found at $xmp" >&2; exit 3; }
md5="$(find "$xmp/third-party/zuid" -type f -name MD5.cpp | head -n1)"
[ -n "$md5" ] || { echo "zuid MD5.cpp not found in XMP toolkit" >&2; exit 3; }
apt-get update -qq >/dev/null
apt-get install -y -qq g++ cmake libjpeg-dev zlib1g-dev libexpat1-dev >/dev/null
jxl="$PWD/jxl"
cmake -S "$root/libjxl/libjxl" -B "$jxl" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF \
  -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF \
  -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF \
  -DJPEGXL_ENABLE_TCMALLOC=OFF -DJPEGXL_ENABLE_PLUGINS=OFF -DJPEGXL_ENABLE_FUZZERS=OFF >/dev/null
cmake --build "$jxl" -j"$(nproc)" --target jxl-static jxl_threads-static >/dev/null
# ExpatAdapter.cpp includes "third-party/expat/lib/expat.h"; the SDK ships no expat sources.
mkdir -p shim/third-party/expat/lib obj
printf '#include <expat.h>\n' >shim/third-party/expat/lib/expat.h
xmp_flags="-std=gnu++11 -O2 -w -fPIC -DUNIX_ENV=1 -DXMP_StaticBuild=1 -DAdobePrivate=0 -DBUILDING_XMPCORE_LIB=1"
xmp_flags="$xmp_flags -DXMP_COMPONENT_INT_NAMESPACE=AdobeXMPCore_Int -I$xmp -I$xmp/public/include -I$PWD/shim"
dng_flags="-std=c++17 -O2 -w -fPIC -DqLinux=1 -DUNIX_ENV=1 -DXMP_StaticBuild=1 -DAdobePrivate=0 -DqDNGUseLibJPEG=1"
dng_flags="$dng_flags -DqDNGUseXMP=1 -DqDNGXMPFiles=0 -DqDNGXMPDocOps=0 -DqDNGThreadSafe=1"
dng_flags="$dng_flags -DqDNGValidateTarget=1 -I$src -I$xmp/public/include"
dng_flags="$dng_flags -I$root/libjxl/libjxl/lib/include -I$jxl/lib/include"
compile() { g++ $FLAGS -c "$1" -o "obj/$(printf %s "$1" | tr / _).o"; }
export -f compile
xmp_dirs=("$xmp/XMPCore/source")
for dir in "$xmp/XMPCommon/source" "$xmp/public/include/XMPCommon/source" "$xmp/public/include/XMPCore/source"; do
  [ ! -d "$dir" ] || xmp_dirs+=("$dir")
done
export FLAGS="$xmp_flags"
{
  find "${xmp_dirs[@]}" -maxdepth 1 -name '*.cpp' ! -name '*DocOps*' -print0
  printf '%s\0' "$xmp/source/UnicodeConversions.cpp" "$xmp/source/XML_Node.cpp" "$xmp/source/XMP_LibUtils.cpp" "$md5"
} | xargs -0 -n1 -P"$(nproc)" bash -c 'compile "$0"'
export FLAGS="$dng_flags"
find "$src" -maxdepth 1 -name '*.cpp' -print0 | xargs -0 -n1 -P"$(nproc)" bash -c 'compile "$0"'
g++ obj/*.o -o dng_validate "$jxl/lib/libjxl.a" "$jxl/lib/libjxl_threads.a" \
  "$jxl"/third_party/brotli/libbrotli{enc,dec,common}-static.a "$jxl/third_party/highway/libhwy.a" \
  -lexpat -ljpeg -lz -lpthread
test -x dng_validate
