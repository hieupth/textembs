use pyo3::prelude::*;
mod tokenizer;


#[pymodule]
#[pyo3(name = "textembserve")]
fn my_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::RustTokenizer>()?;
    Ok(())
}
