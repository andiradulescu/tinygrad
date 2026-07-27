import os, pathlib, platform, subprocess, sys, unittest

TINYGRAD_ROOT = pathlib.Path(__file__).parents[2]

@unittest.skipUnless(platform.machine().lower() in {"arm64", "aarch64"} and pathlib.Path("/dev/kgsl-3d0").exists(), "requires KGSL")
class TestQCOMSniffer(unittest.TestCase):
  def test_imports_with_current_autogen_and_kgsl_structs(self):
    src = ("from extra.qcom_gpu_driver.opencl_ioctl import format_struct; "
           "from extra.qcom_gpu_driver.msm_kgsl import struct_kgsl_syncsource_create; "
           "format_struct(struct_kgsl_syncsource_create())")
    result = subprocess.run([sys.executable, "-c", src], cwd=TINYGRAD_ROOT, env={**os.environ, "IOCTL": "1"},
                            capture_output=True, text=True)

    self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
  unittest.main()
