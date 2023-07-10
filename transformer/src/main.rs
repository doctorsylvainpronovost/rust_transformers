extern crate ndarray;
use ndarray::prelude::*;
// use ndarray::Array1;

fn main() {
    // Example usage
    let tsfmr = transformer::Transformer::new(4, 512, 2048);
    let input = Array2::zeros((16, 512));
    let output = tsfmr.forward(input);

    println!("Output shape: {:?}", output.shape());
}