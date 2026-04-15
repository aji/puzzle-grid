use std::borrow::Borrow;

use puzzle_grid::array::Array;

const ARRAY_BUFFER_2X3: [usize; 6] = [1, 2, 3, 4, 5, 6];
const ARRAY_BUFFER_4X4: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

fn array2x3() -> Array<usize, [usize; 6]> {
    Array::new(2, 3, ARRAY_BUFFER_2X3)
}
fn array4x4() -> Array<usize, [usize; 16]> {
    Array::new(4, 4, ARRAY_BUFFER_4X4)
}

fn row2<B: AsRef<[usize]>>(arr: impl Borrow<Array<usize, B>>, row: usize) -> [usize; 2] {
    let arr = arr.borrow();
    [arr[(row, 0)], arr[(row, 1)]]
}
fn row3<B: AsRef<[usize]>>(arr: impl Borrow<Array<usize, B>>, row: usize) -> [usize; 3] {
    let arr = arr.borrow();
    [arr[(row, 0)], arr[(row, 1)], arr[(row, 2)]]
}
fn row4<B: AsRef<[usize]>>(arr: impl Borrow<Array<usize, B>>, row: usize) -> [usize; 4] {
    let arr = arr.borrow();
    [arr[(row, 0)], arr[(row, 1)], arr[(row, 2)], arr[(row, 3)]]
}

fn is_borrowed<'a, T: Clone>(cow: &std::borrow::Cow<'a, [T]>) -> bool {
    match cow {
        std::borrow::Cow::Borrowed(_) => true,
        std::borrow::Cow::Owned(_) => false,
    }
}

#[test]
pub fn test_array() {
    let arr = array2x3();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [1, 2, 3]);
    assert_eq!(row3(arr, 1), [4, 5, 6]);
}

#[test]
pub fn test_array_transpose() {
    let arr = array2x3().transpose();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [1, 4]);
    assert_eq!(row2(arr, 1), [2, 5]);
    assert_eq!(row2(arr, 2), [3, 6]);
}

#[test]
pub fn test_array_flip_h() {
    let arr = array2x3().flip_h();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [3, 2, 1]);
    assert_eq!(row3(arr, 1), [6, 5, 4]);
}

#[test]
pub fn test_array_flip_v() {
    let arr = array2x3().flip_v();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [4, 5, 6]);
    assert_eq!(row3(arr, 1), [1, 2, 3]);
}

#[test]
pub fn test_array_rotate_cw() {
    let arr = array2x3().rotate_cw();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [4, 1]);
    assert_eq!(row2(arr, 1), [5, 2]);
    assert_eq!(row2(arr, 2), [6, 3]);
}

#[test]
pub fn test_array_rotate_180() {
    let arr = array2x3().rotate_180();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [6, 5, 4]);
    assert_eq!(row3(arr, 1), [3, 2, 1]);
}

#[test]
pub fn test_array_rotate_ccw() {
    let arr = array2x3().rotate_ccw();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [3, 6]);
    assert_eq!(row2(arr, 1), [2, 5]);
    assert_eq!(row2(arr, 2), [1, 4]);
}

#[test]
pub fn test_array_view() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3);
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [10, 11, 12]);
    assert_eq!(row3(arr, 1), [14, 15, 16]);
}

#[test]
pub fn test_array_view_transpose() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).transpose();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [10, 14]);
    assert_eq!(row2(arr, 1), [11, 15]);
    assert_eq!(row2(arr, 2), [12, 16]);
}

#[test]
pub fn test_array_view_flip_h() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).flip_h();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [12, 11, 10]);
    assert_eq!(row3(arr, 1), [16, 15, 14]);
}

#[test]
pub fn test_array_view_flip_v() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).flip_v();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [14, 15, 16]);
    assert_eq!(row3(arr, 1), [10, 11, 12]);
}

#[test]
pub fn test_array_view_rotate_cw() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).rotate_cw();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [14, 10]);
    assert_eq!(row2(arr, 1), [15, 11]);
    assert_eq!(row2(arr, 2), [16, 12]);
}

#[test]
pub fn test_array_view_rotate_180() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).rotate_180();
    assert_eq!(arr.shape(), (2, 3));
    assert_eq!(row3(arr, 0), [16, 15, 14]);
    assert_eq!(row3(arr, 1), [12, 11, 10]);
}

#[test]
pub fn test_array_view_rotate_ccw() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).rotate_ccw();
    assert_eq!(arr.shape(), (3, 2));
    assert_eq!(row2(arr, 0), [12, 16]);
    assert_eq!(row2(arr, 1), [11, 15]);
    assert_eq!(row2(arr, 2), [10, 14]);
}

#[test]
pub fn test_array_mut() {
    let mut arr = array2x3();
    arr[0] = 9;
    assert_eq!(row3(arr, 0), [9, 2, 3]);
    assert_eq!(row3(arr, 1), [4, 5, 6]);
    arr[(1, 1)] = 0;
    assert_eq!(row3(arr, 0), [9, 2, 3]);
    assert_eq!(row3(arr, 1), [4, 0, 6]);
}

#[test]
pub fn test_array_view_mut() {
    let mut arr = array4x4();

    let _ = {
        let mut arr = arr.view_mut(2, 1, 2, 3);
        arr[0] = 99;
        assert_eq!(row3(&arr, 0), [99, 11, 12]);
        assert_eq!(row3(&arr, 1), [14, 15, 16]);
        arr[(1, 1)] = 98;
        assert_eq!(row3(&arr, 0), [99, 11, 12]);
        assert_eq!(row3(&arr, 1), [14, 98, 16]);
    };

    assert_eq!(row4(&arr, 0), [1, 2, 3, 4]);
    assert_eq!(row4(&arr, 1), [5, 6, 7, 8]);
    assert_eq!(row4(&arr, 2), [9, 99, 11, 12]);
    assert_eq!(row4(&arr, 3), [13, 14, 98, 16]);
}

#[test]
pub fn test_array_view_mut_flip_h() {
    let mut arr = array4x4();

    let _ = {
        let mut arr = arr.view_mut(2, 1, 2, 3).flip_h();
        arr[0] = 99;
        assert_eq!(row3(&arr, 0), [99, 11, 10]);
        assert_eq!(row3(&arr, 1), [16, 15, 14]);
        arr[(1, 1)] = 98;
        assert_eq!(row3(&arr, 0), [99, 11, 10]);
        assert_eq!(row3(&arr, 1), [16, 98, 14]);
    };

    assert_eq!(row4(&arr, 0), [1, 2, 3, 4]);
    assert_eq!(row4(&arr, 1), [5, 6, 7, 8]);
    assert_eq!(row4(&arr, 2), [9, 10, 11, 99]);
    assert_eq!(row4(&arr, 3), [13, 14, 98, 16]);
}

#[test]
pub fn test_array_iter() {
    let arr = array2x3();
    let res: Vec<usize> = arr.iter().copied().collect();
    assert_eq!(res, &[1, 2, 3, 4, 5, 6]);
}

#[test]
pub fn test_array_view_iter() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3);
    let res: Vec<usize> = arr.iter().copied().collect();
    assert_eq!(res, &[10, 11, 12, 14, 15, 16]);
}

#[test]
pub fn test_array_flip_h_iter() {
    let arr = array2x3().flip_h();
    let res: Vec<usize> = arr.iter().copied().collect();
    assert_eq!(res, &[3, 2, 1, 6, 5, 4]);
}

#[test]
pub fn test_array_view_flip_h_iter() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3).flip_h();
    let res: Vec<usize> = arr.iter().copied().collect();
    assert_eq!(res, &[12, 11, 10, 16, 15, 14]);
}

#[test]
pub fn test_array_iter_mut() {
    let mut arr = array2x3();
    arr.iter_mut().enumerate().for_each(|(i, x)| *x *= i);
    assert_eq!(row3(arr, 0), [0, 2, 6]);
    assert_eq!(row3(arr, 1), [12, 20, 30]);
}

#[test]
pub fn test_array_view_iter_mut() {
    let mut arr = array4x4();
    arr.view_mut(2, 1, 2, 3)
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x *= i);
    assert_eq!(row4(&arr, 0), [1, 2, 3, 4]);
    assert_eq!(row4(&arr, 1), [5, 6, 7, 8]);
    assert_eq!(row4(&arr, 2), [9, 0, 11, 24]);
    assert_eq!(row4(&arr, 3), [13, 42, 60, 80]);
}

#[test]
pub fn test_array_from_iter() {
    let arr = (1..=6)
        .collect::<Array<usize, Vec<usize>>>()
        .reshape(2, 3)
        .unwrap();
    assert_eq!(row3(&arr, 0), [1, 2, 3]);
    assert_eq!(row3(&arr, 1), [4, 5, 6]);
}

#[test]
pub fn test_array_reshape_ok() {
    let arr = array2x3().reshape(3, 2).unwrap();
    assert_eq!(row2(arr, 0), [1, 2]);
    assert_eq!(row2(arr, 1), [3, 4]);
    assert_eq!(row2(arr, 2), [5, 6]);
}

#[test]
pub fn test_array_reshape_bad() {
    assert!(array2x3().reshape(2, 2).is_none());
}

#[test]
pub fn test_array_as_contiguous() {
    let arr = array4x4();
    assert!(is_borrowed(&arr.as_contiguous().unwrap()));
    assert!(is_borrowed(&arr.view(2, 0, 2, 4).as_contiguous().unwrap()));
    assert!(is_borrowed(&arr.view(2, 1, 1, 3).as_contiguous().unwrap()));
    assert!(!is_borrowed(&arr.view(2, 0, 2, 3).as_contiguous().unwrap()));
    assert!(!is_borrowed(&arr.view(2, 1, 2, 3).as_contiguous().unwrap()));
}

#[test]
pub fn test_array_non_contiguous_reshape() {
    let arr = array4x4();
    let arr = arr.view(2, 1, 2, 3);

    assert!(arr.reshape(1, 6).is_none());
    assert!(arr.as_contiguous().reshape(1, 6).is_some());
}
