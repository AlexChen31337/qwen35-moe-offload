fn main() {
    let venv = "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages";
    let lib_dir = format!("{}/llama_cpp/lib", venv);
    let include_dir = format!("{}/include", venv);

    // Compile the C shim
    cc::Build::new()
        .file("shim.c")
        .include(&include_dir)
        .compile("bench_shim");

    // Link paths
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-lib=dylib=ggml-cuda");

    // RPATH so the binary finds the .so at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    
    // Rerun if shim changes
    println!("cargo:rerun-if-changed=shim.c");
}
