use puzzle_grid::array::Array;

const ARRAY_BUFFER: &'static [usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];

fn array() -> Array<usize, &'static [usize]> {
    Array::new(3, 3, ARRAY_BUFFER)
}

#[test]
pub fn test_array_transpose() {
    let arr = array().transpose();

    // 0 1 2    0 3 6
    // 3 4 5 -> 1 4 7
    // 6 7 8    2 5 8

    assert_eq!(arr[(0, 0)], 0);
    assert_eq!(arr[(0, 1)], 3);
    assert_eq!(arr[(0, 2)], 6);

    assert_eq!(arr[(1, 0)], 1);
    assert_eq!(arr[(1, 1)], 4);
    assert_eq!(arr[(1, 2)], 7);

    assert_eq!(arr[(2, 0)], 2);
    assert_eq!(arr[(2, 1)], 5);
    assert_eq!(arr[(2, 2)], 8);
}
