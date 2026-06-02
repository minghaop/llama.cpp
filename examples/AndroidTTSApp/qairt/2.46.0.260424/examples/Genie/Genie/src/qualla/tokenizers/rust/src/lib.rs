//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** [lib.rs]
 *  Rust implementation of the C-wrapper headers in tokenizers-capi.h using
 *  the huggingface/tokenizer crate
 */

use std::path::Path;
use std::str::FromStr;
use tokenizers::tokenizer::Tokenizer;
#[cfg(feature = "mmap")]
use memmap2::Mmap;
#[cfg(feature = "mmap")]
use std::fs::File;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    encode_ids: Vec<u32>,
    decode_str: String,
}

impl TokenizerWrapper {
    pub fn from_str(json: &str) -> Result<TokenizerWrapper, String> {
        let tokenizer = Tokenizer::from_str(json).map_err(|e| format!("Failed to create tokenizer: {}",e))?;
        Ok (TokenizerWrapper {
            tokenizer,
            encode_ids: Vec::new(),
            decode_str: String::new(),
        })
    }

    #[cfg(feature = "mmap")]
    pub fn from_file(path: &Path) -> Result<TokenizerWrapper, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| format!("Failed to mmap file: {}", e))?
        };
        let json_str = std::str::from_utf8(&mmap).map_err(|e| format!("Invalid UTF-8 in file: {}", e))?;
        let tokenizer = Tokenizer::from_str(json_str).map_err(|e| format!("Failed to create tokenizer: {}", e))?;
        Ok(TokenizerWrapper {
            tokenizer,
            encode_ids: Vec::new(),
            decode_str: String::new(),
        })
    }

    #[cfg(not(feature = "mmap"))]
    pub fn from_file(path: &Path) -> Result<TokenizerWrapper, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        Self::from_str(&json_str)
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) {
        self.encode_ids = Vec::from(self.tokenizer.encode(text, add_special_tokens).unwrap().get_ids());
    }

    pub fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
unsafe extern "C" fn tokenizers_new_from_str(input_cstr: *const u8,
                                             len: usize) -> *mut TokenizerWrapper {
    let json = match std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match TokenizerWrapper::from_str(json) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
unsafe extern "C" fn tokenizers_new_from_file(path_cstr: *const u8,
                                             len: usize) -> *mut TokenizerWrapper {
    let path_cstr = match std::str::from_utf8(std::slice::from_raw_parts(path_cstr, len)) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let path = Path::new(path_cstr);
    match TokenizerWrapper::from_file(path) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
unsafe extern "C" fn tokenizers_encode(handle: *mut TokenizerWrapper,
                                       input_cstr: *const u8,
                                       len: usize,
                                       add_special_tokens: i32) {
    let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
    (*handle).encode(input_data, add_special_tokens != 0);
}

#[no_mangle]
unsafe extern "C" fn tokenizers_get_encode_ids(handle: *mut TokenizerWrapper,
                                               out_data: *mut *mut u32,
                                               out_len: *mut usize) {
    *out_data = (*handle).encode_ids.as_mut_ptr();
    *out_len  = (&(*handle).encode_ids).len();
}

#[no_mangle]
unsafe extern "C" fn tokenizers_decode(handle: *mut TokenizerWrapper,
                                       input_ids: *const u32,
                                       len: usize,
                                       skip_special_tokens: i32) {
    let input_data = std::slice::from_raw_parts(input_ids, len);
    (*handle).decode(input_data, skip_special_tokens != 0);
}

#[no_mangle]
unsafe extern "C" fn tokenizers_get_decode_str(handle: *mut TokenizerWrapper,
                                               out_cstr: *mut *mut u8,
                                               out_len: *mut usize) {
    *out_cstr = (*handle).decode_str.as_mut_ptr();
    *out_len  = (&(*handle).decode_str).len();
}

#[no_mangle]
unsafe extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    drop(Box::from_raw(wrapper));
}

#[link(name = "onig")]
extern "C" {
    fn onig_end();
}

#[no_mangle]
pub extern "C" fn tokenizer_cleanup() {
    unsafe {
        onig_end();
    }
}
