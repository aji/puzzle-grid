use puzzle_grid::grid::GridBuilder;

#[test]
pub fn test_grid_full_layer() {
    let grid = GridBuilder::new(3, 3).pad(1, 0, 0, 2).build();
    let layer = grid.new_full_layer::<usize>();
    let cells = layer.cells();
    assert_eq!(cells.shape(), (4, 5));
}

#[test]
pub fn test_grid_grid_layer() {
    let grid = GridBuilder::new(3, 3).pad(1, 0, 0, 2).build();
    let layer = grid.new_grid_layer::<usize>();
    let cells = layer.cells();
    assert_eq!(cells.shape(), (3, 3));
}
