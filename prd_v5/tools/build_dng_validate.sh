#!/usr/bin/env bash
# Best-effort Linux build of Adobe's dng_validate. Usage: build_dng_validate.sh WORKDIR URL SHA256
# The SDK ships Xcode and Visual Studio projects only, so this compiles the DNG SDK sources and the
# bundled XMPCore toolkit (XMPCore, XMPCommon, shared sources and zuid MD5, as in the toolkit's own
# XMPCore CMake list) directly, against system expat, zlib and libjpeg, with JPEG XL support off.
# It is unverified until a Colab run; any failure exits nonzero and the caller records a skip.
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
apt-get install -y -qq g++ libjpeg-dev zlib1g-dev libexpat1-dev >/dev/null
# ExpatAdapter.cpp includes "third-party/expat/lib/expat.h"; the SDK ships no expat sources.
mkdir -p shim/third-party/expat/lib obj
printf '#include <expat.h>\n' >shim/third-party/expat/lib/expat.h
xmp_flags="-std=gnu++11 -O2 -w -fPIC -DUNIX_ENV=1 -DXMP_StaticBuild=1 -DAdobePrivate=0 -DBUILDING_XMPCORE_LIB=1"
xmp_flags="$xmp_flags -DXMP_COMPONENT_INT_NAMESPACE=AdobeXMPCore_Int -I$xmp -I$xmp/public/include -I$PWD/shim"
dng_flags="-std=c++17 -O2 -w -fPIC -DqLinux=1 -DUNIX_ENV=1 -DXMP_StaticBuild=1 -DqDNGUseLibJPEG=1"
dng_flags="$dng_flags -DqDNGSupportJXL=0 -DqDNGUseXMP=1 -DqDNGXMPFiles=0 -DqDNGXMPDocOps=0 -DqDNGThreadSafe=1"
dng_flags="$dng_flags -DqDNGValidateTarget=1 -I$src -I$xmp/public/include"
compile() { g++ $FLAGS -c "$1" -o "obj/$(printf %s "$1" | tr / _).o"; }
export -f compile
xmp_dirs=("$xmp/XMPCore/source")
[ ! -d "$xmp/XMPCommon/source" ] || xmp_dirs+=("$xmp/XMPCommon/source")
export FLAGS="$xmp_flags"
{
  find "${xmp_dirs[@]}" -maxdepth 1 -name '*.cpp' -print0
  printf '%s\0' "$xmp/source/UnicodeConversions.cpp" "$xmp/source/XML_Node.cpp" "$xmp/source/XMP_LibUtils.cpp" "$md5"
} | xargs -0 -n1 -P"$(nproc)" bash -c 'compile "$0"'
export FLAGS="$dng_flags"
find "$src" -maxdepth 1 -name '*.cpp' -print0 | xargs -0 -n1 -P"$(nproc)" bash -c 'compile "$0"'
g++ obj/*.o -o dng_validate -lexpat -ljpeg -lz -lpthread
test -x dng_validate
