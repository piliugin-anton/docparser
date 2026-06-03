//! When built with `--features mkl` and `MKLROOT` set, embed rpath entries for MKL and
//! Intel OpenMP (`libiomp5.so`) so the release binary runs without sourcing oneAPI env.

fn main() {
    if std::env::var_os("CARGO_FEATURE_MKL").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=MKLROOT");

    let Ok(mklroot) = std::env::var("MKLROOT") else {
        println!(
            "cargo:warning=MKLROOT is not set. Link against system MKL and set MKLROOT when building with --features mkl, or install intel-oneapi-openmp and source compiler vars at runtime."
        );
        return;
    };

    let mkl_lib = format!("{mklroot}/lib");
    if std::path::Path::new(&mkl_lib).is_dir() {
        println!("cargo:rustc-link-search=native={mkl_lib}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{mkl_lib}");
    }

    // MKL linked with Intel OpenMP (lp64-iomp) needs libiomp5 from the oneAPI compiler.
    let mklroot_path = std::path::Path::new(&mklroot);
    if let Some(oneapi) = mklroot_path.parent().and_then(|p| p.parent()) {
        let compiler_lib = oneapi.join("compiler/latest/lib");
        if compiler_lib.is_dir() {
            let path = compiler_lib.to_string_lossy();
            println!("cargo:rustc-link-search=native={path}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
        }
    }
}
