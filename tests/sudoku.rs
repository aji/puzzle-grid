use puzzle_grid::{
    array::ArrayView,
    grid::{Grid, GridBuilder, LayerVec},
    iter::IteratorExt,
};

struct SudokuPuzzle {
    #[allow(unused)]
    grid: Grid,
    digits: LayerVec<Option<u8>>,
}

type SudokuSet<'a> = ArrayView<'a, Option<u8>>;

impl SudokuPuzzle {
    fn new() -> SudokuPuzzle {
        let grid = GridBuilder::new(9, 9).build();
        let digits = grid.new_grid_layer();
        SudokuPuzzle { grid, digits }
    }

    fn row(&'_ self, i: usize) -> SudokuSet<'_> {
        self.digits.cells().into_row(i)
    }

    fn col(&'_ self, i: usize) -> SudokuSet<'_> {
        self.digits.cells().into_col(i)
    }

    fn house(&'_ self, i: usize) -> SudokuSet<'_> {
        let r = i / 3;
        let c = i % 3;
        self.digits.cells().into_view(r * 3, c * 3, 3, 3)
    }

    fn is_valid(&self) -> bool {
        for i in 0..9 {
            let row = is_set_valid(self.row(i));
            let col = is_set_valid(self.col(i));
            let house = is_set_valid(self.house(i));
            if !row || !col || !house {
                return false;
            }
        }
        true
    }
}

fn is_set_valid(set: SudokuSet) -> bool {
    let unique = set
        .iter()
        .copied()
        .flat_map(|x| x.into_iter())
        .filter(|x| 1 <= *x && *x <= 9)
        .collect::<std::collections::HashSet<u8>>()
        .len();
    unique == 9
}

const GRID: [[u8; 9]; 9] = [
    [5, 3, 4, /**/ 6, 7, 8, /**/ 9, 1, 2],
    [6, 7, 2, /**/ 1, 9, 5, /**/ 3, 4, 8],
    [1, 9, 8, /**/ 3, 4, 2, /**/ 5, 6, 7],
    /************************************/
    [8, 5, 9, /**/ 7, 6, 1, /**/ 4, 2, 3],
    [4, 2, 6, /**/ 8, 5, 3, /**/ 7, 9, 1],
    [7, 1, 3, /**/ 9, 2, 4, /**/ 8, 5, 6],
    /************************************/
    [9, 6, 1, /**/ 5, 3, 7, /**/ 2, 8, 4],
    [2, 8, 7, /**/ 4, 1, 9, /**/ 6, 3, 5],
    [3, 4, 5, /**/ 2, 8, 6, /**/ 1, 7, 9],
];

#[test]
fn test_sudoku() {
    let mut puz = SudokuPuzzle::new();

    assert!(!puz.is_valid());

    puz.digits = GRID
        .as_flattened()
        .iter()
        .map(|x| Some(*x))
        .into_grid_layer(&puz.grid);

    assert!(puz.is_valid());
}
