use pyo3::types::*;
use pyo3::prelude::*;
use pyo3::exceptions::*;
use tokenizers::tokenizer::{Tokenizer};

#[pyclass]
pub struct RustTokenizer
{
  _tokenizer: Tokenizer
}

#[pymethods]
impl RustTokenizer
{
  #[new]
  pub fn new(model: &str) -> PyResult<Self> {
    return match Tokenizer::from_pretrained(model, None) {
      Ok(t) => Ok(RustTokenizer{ _tokenizer: t}),
      Err(_) => Err(PyRuntimeError::new_err("RustTokenizer could not initialized!"))
    };
  }

  pub fn encode<'a>(&'a self, inputs: Vec<&str>, py: Python<'a>) -> PyResult<&PyDict> {
    return match self._tokenizer.encode_batch(inputs, false) {
      Ok(_encoding) => {
        let input_ids: Vec<_> = _encoding.iter().map(|x| x.get_ids()).collect();
        let token_type_ids: Vec<_> = _encoding.iter().map(|x| x.get_type_ids()).collect();
        let attention_mask: Vec<_> = _encoding.iter().map(|x| x.get_attention_mask()).collect();
        return Ok([
          ("input_ids", input_ids),
          ("token_type_ids", token_type_ids),
          ("attention_mask", attention_mask)
        ].into_py_dict(py));
      },
      Err(_) => Err(PyRuntimeError::new_err("RustTokenizer could not encode inputs!"))
    };
  }
}