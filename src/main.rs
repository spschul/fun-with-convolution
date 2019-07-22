#![feature(test)]
extern crate test;

use rayon::prelude::*;

pub struct Image<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
    // perhaps stride might be necessary in the real world??? not sure
}

impl<T> Image<T> {
    fn get_pixel(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    fn get_pixel_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }
}

pub fn naive_convolve(src: &Image<i32>, ker: &Image<i32>) -> Image<i32> {
    let (out_rows, out_cols) = (src.rows - ker.rows + 1, src.cols - ker.cols + 1);
    let mut output: Image<i32> = Image {
        rows: out_rows,
        cols: out_cols,
        data: vec![0; (out_rows * out_cols) as usize],
    };
    for src_r in 0..(src.rows - ker.rows + 1) {
        for src_c in 0..(src.cols - ker.cols + 1) {
            for ker_r in 0..ker.rows {
                for ker_c in 0..ker.cols {
                    *output.get_pixel_mut(src_r, src_c) +=
                        *src.get_pixel(src_r + ker_r, src_c + ker_c) * *ker.get_pixel(ker_r, ker_c);
                }
            }
        }
    }
    output
}

pub fn convolve_simple_skip_functions(src: &Image<i32>, ker: &Image<i32>) -> Image<i32> {
    let (out_rows, out_cols) = (src.rows - ker.rows + 1, src.cols - ker.cols + 1);
    let mut output: Image<i32> = Image {
        rows: out_rows,
        cols: out_cols,
        data: vec![0; (out_rows * out_cols) as usize],
    };
    for src_r in 0..(src.rows - ker.rows + 1) {
        for src_c in 0..(src.cols - ker.cols + 1) {
            for ker_r in 0..ker.rows {
                for ker_c in 0..ker.cols {
                    output.data[src_r * output.cols + src_c] += src.data
                        [(src_r + ker_r) * src.cols + src_c + ker_c]
                        * ker.data[ker_r * ker.cols + ker_c];
                }
            }
        }
    }
    output
}

pub fn convolve_some_slices(src: &Image<i32>, ker: &Image<i32>) -> Image<i32> {
    let (out_rows, out_cols) = (src.rows - ker.rows + 1, src.cols - ker.cols + 1);
    let mut output: Image<i32> = Image {
        rows: out_rows,
        cols: out_cols,
        data: vec![0; (out_rows * out_cols) as usize],
    };
    for (out_row, out_row_index) in output
        .data
        .chunks_exact_mut(output.cols)
        .into_iter()
        .zip(0..output.rows)
    {
        for out_col_index in 0..out_row.len() {
            for ker_r in 0..ker.rows {
                for ker_c in 0..ker.cols {
                    out_row[out_col_index] +=
                        src.data[(out_row_index + ker_r) * src.cols + out_col_index + ker_c];
                }
            }
        }
    }
    output
}

pub fn convolve_some_slices_par(src: &Image<i32>, ker: &Image<i32>) -> Image<i32> {
    let (out_rows, out_cols) = (src.rows - ker.rows + 1, src.cols - ker.cols + 1);
    let mut output: Image<i32> = Image {
        rows: out_rows,
        cols: out_cols,
        data: vec![0; (out_rows * out_cols) as usize],
    };
    let mut output_rows: Vec<(&mut [i32], usize)> = output
        .data
        .chunks_exact_mut(output.cols)
        .zip(0..output.rows)
        .collect();
    output_rows
        .par_iter_mut()
        .for_each(|(out_row, out_row_index)| {
            for out_col_index in 0..out_row.len() {
                for ker_r in 0..ker.rows {
                    for ker_c in 0..ker.cols {
                        out_row[out_col_index] +=
                            src.data[(*out_row_index + ker_r) * src.cols + out_col_index + ker_c];
                    }
                }
            }
        });
    output
}

pub fn just_zero_fill(src: &Image<i32>, ker: &Image<i32>) -> Image<i32> {
    let (out_rows, out_cols) = (src.rows - ker.rows + 1, src.cols - ker.cols + 1);
    Image {
        rows: out_rows,
        cols: out_cols,
        data: vec![0; (out_rows * out_cols) as usize],
    }
}

pub fn get_image_and_kernel() -> (Image<i32>, Image<i32>) {
    // 720p
    // let (rows, cols) = (1280, 720);
    // 4K
    // let (rows, cols) = (3840, 2160);
    // 8K
    // let (rows, cols): (usize, usize) = (7680, 4320);
    // Cruel
    let (rows, cols): (usize, usize) = (7680 * 4, 4320 * 4);
    let data: Vec<i32> = (0..(rows as i32 * cols as i32)).map(|x| x % 100).collect();
    let to_convolve: Image<i32> = Image {
        rows: rows,
        cols: cols,
        data: data,
    };
    let kernel_size = 3;
    let kernel_data: Vec<i32> = (0..(kernel_size as i32 * kernel_size as i32)).collect();
    let kernel: Image<i32> = Image {
        rows: kernel_size,
        cols: kernel_size,
        data: kernel_data,
    };
    (to_convolve, kernel)
}

fn main() {
    let (image, kernel) = get_image_and_kernel();
    let conv = naive_convolve(&image, &kernel);
    // let conv = convolve_some_slices_par(&image, &kernel);
    println!("{}", conv.data.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn test_naive_convolve(b: &mut Bencher) {
        let (image, kernel) = get_image_and_kernel();
        b.iter(|| {
            naive_convolve(&image, &kernel);
        });
    }

    #[bench]
    fn test_convolve_simple_skip_functions(b: &mut Bencher) {
        let (image, kernel) = get_image_and_kernel();
        b.iter(|| {
            convolve_simple_skip_functions(&image, &kernel);
        });
    }

    #[bench]
    fn test_convolve_some_slices(b: &mut Bencher) {
        let (image, kernel) = get_image_and_kernel();
        b.iter(|| {
            convolve_some_slices(&image, &kernel);
        });
    }

    #[bench]
    fn test_convolve_some_slices_par(b: &mut Bencher) {
        let (image, kernel) = get_image_and_kernel();
        b.iter(|| {
            convolve_some_slices_par(&image, &kernel);
        });
    }

    #[bench]
    fn test_just_zero_fill(b: &mut Bencher) {
        let (image, kernel) = get_image_and_kernel();
        b.iter(|| {
            just_zero_fill(&image, &kernel);
        });
    }
}
