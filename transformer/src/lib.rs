extern crate ndarray;
use ndarray::prelude::*;
use ndarray::Array1;

#[derive(Debug)]
pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize) -> Self {
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;

        // Initialize weights
        let w_q = Array2::from_elem((d_model, d_k), 0.0);
        let w_k = Array2::from_elem((d_model, d_k), 0.0);
        let w_v = Array2::from_elem((d_model, d_v), 0.0);
        let w_o = Array2::from_elem((d_model, d_model), 0.0);

        MultiHeadAttention {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        // Perform self-attention
        // TODO: Implement self-attention logic

        // Apply output linear layer
        let y = x.dot(&self.w_o);

        y
    }
}

#[derive(Debug)]
pub struct FeedForward {
    d_model: usize,
    d_ff: usize,
    w_1: Array2<f32>,
    w_2: Array2<f32>,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        // Initialize weights
        let w_1 = Array2::from_elem((d_model, d_ff), 0.0);
        let w_2 = Array2::from_elem((d_ff, d_model), 0.0);

        FeedForward {
            d_model,
            d_ff,
            w_1,
            w_2,
        }
    }

    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        // Apply first linear layer
        let h = x.dot(&self.w_1);

        // Apply activation function (e.g., ReLU)
        let h_relu = h.mapv(|v| if v > 0.0 { v } else { 0.0 });

        // Apply second linear layer
        let y = h_relu.dot(&self.w_2);

        y
    }
}

#[derive(Debug)]
pub struct Transformer {
    num_heads: usize,
    d_model: usize,
    d_ff: usize,
    attn: MultiHeadAttention,
    ff: FeedForward,
}

impl Transformer {
    pub fn new(num_heads: usize, d_model: usize, d_ff: usize) -> Self {
        let attn = MultiHeadAttention::new(num_heads, d_model);
        let ff = FeedForward::new(d_model, d_ff);

        Transformer {
            num_heads,
            d_model,
            d_ff,
            attn,
            ff,
        }
    }

    pub fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        // Apply self-attention
        let attn_output = self.attn.forward(x.clone());

        // Apply residual connection and layer normalization
        let residual = x + attn_output;
        let norm1 = Self::layer_norm(residual);

        // Apply feed-forward layers
        let ff_output = self.ff.forward(norm1.clone());

        // Apply residual connection and layer normalization
        let output = norm1 + ff_output;
        let norm2 = Self::layer_norm(output);

        norm2
    }

    pub fn layer_norm(x: Array2<f32>) -> Array2<f32> {
        let epsilon = 1e-6; // Small constant to avoid division by zero
    
        let num_rows = x.shape()[0];
        let num_cols = x.shape()[1];
    
        // Compute mean and variance along the last axis (columns)
        let mean = x.mean_axis(Axis(1)).unwrap(); // Compute mean along columns
        let variance = x.var_axis(Axis(1), 0.0); // Compute variance along columns
    
        // Normalize the input
        let mut normalized = Array2::from_elem(x.dim(), 0.0);
        for i in 0..num_rows {
            for j in 0..num_cols {
                normalized[[i, j]] = (x[[i, j]] - mean[i]) / (variance[i] + epsilon).sqrt();
            }
        }
    
        // Scale and shift the normalized values
        let gamma = Array1::from_elem(num_cols, 1.0); // Scaling factor
        let beta = Array1::from_elem(num_cols, 0.0); // Shifting factor
        let scaled_shifted = normalized * &gamma + &beta;
    
        scaled_shifted
    }

}

fn main() {
    // Example usage
    let tsfmr = Transformer::new(4, 512, 2048);
    let input = Array2::zeros((16, 512));
    let output = tsfmr.forward(input);

    println!("Output shape: {:?}", output.shape());
}
