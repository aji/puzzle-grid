use puzzle_grid::{array::ArrayBuffer, grid::GridBuilder, iter::IteratorExt};

#[test]
pub fn test_iter_assign_to_array() {
    let arr = {
        let mut arr = ArrayBuffer::new_default(3, 3);
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
            .into_iter()
            .assign_to_array(&mut arr);
        arr
    };
    assert_eq!(arr[(0, 0)], 1);
    assert_eq!(arr[(0, 1)], 2);
    assert_eq!(arr[(0, 2)], 3);
    assert_eq!(arr[(1, 0)], 4);
    assert_eq!(arr[(1, 1)], 5);
    assert_eq!(arr[(1, 2)], 6);
    assert_eq!(arr[(2, 0)], 7);
    assert_eq!(arr[(2, 1)], 8);
    assert_eq!(arr[(2, 2)], 9);
}

#[test]
pub fn test_iter_assign_to_array_short() {
    let arr = {
        let mut arr = ArrayBuffer::new_default(3, 3);
        [1, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .assign_to_array(&mut arr);
        arr
    };
    assert_eq!(arr[(0, 0)], 1);
    assert_eq!(arr[(0, 1)], 2);
    assert_eq!(arr[(0, 2)], 3);
    assert_eq!(arr[(1, 0)], 4);
    assert_eq!(arr[(1, 1)], 5);
    assert_eq!(arr[(1, 2)], 6);
    assert_eq!(arr[(2, 0)], 7);
    assert_eq!(arr[(2, 1)], 8);
    assert_eq!(arr[(2, 2)], 0);
}

#[test]
pub fn test_iter_assign_to_array_long() {
    let arr = {
        let mut arr = ArrayBuffer::new_default(3, 3);
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .into_iter()
            .assign_to_array(&mut arr);
        arr
    };
    assert_eq!(arr[(0, 0)], 1);
    assert_eq!(arr[(0, 1)], 2);
    assert_eq!(arr[(0, 2)], 3);
    assert_eq!(arr[(1, 0)], 4);
    assert_eq!(arr[(1, 1)], 5);
    assert_eq!(arr[(1, 2)], 6);
    assert_eq!(arr[(2, 0)], 7);
    assert_eq!(arr[(2, 1)], 8);
    assert_eq!(arr[(2, 2)], 9);
}

#[test]
pub fn test_iter_into_full_layer() {
    let grid = GridBuilder::new(2, 2).pad(1, 0, 0, 1).build();
    let layer = (1..=9).into_full_layer(&grid);
    let arr = layer.cells();
    assert_eq!(arr.shape(), (3, 3));
    assert_eq!(arr[(0, 0)], 1);
    assert_eq!(arr[(0, 1)], 2);
    assert_eq!(arr[(0, 2)], 3);
    assert_eq!(arr[(1, 0)], 4);
    assert_eq!(arr[(1, 1)], 5);
    assert_eq!(arr[(1, 2)], 6);
    assert_eq!(arr[(2, 0)], 7);
    assert_eq!(arr[(2, 1)], 8);
    assert_eq!(arr[(2, 2)], 9);
}

#[test]
pub fn test_iter_into_grid_layer() {
    let grid = GridBuilder::new(2, 2).pad(1, 0, 0, 1).build();
    let layer = (1..=4).into_grid_layer(&grid);
    let arr = layer.cells();
    assert_eq!(arr.shape(), (2, 2));
    assert_eq!(arr[(0, 0)], 1);
    assert_eq!(arr[(0, 1)], 2);
    assert_eq!(arr[(1, 0)], 3);
    assert_eq!(arr[(1, 1)], 4);
}
