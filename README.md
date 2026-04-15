# `puzzle-grid`

This is a Rust crate with useful data structures for puzzle grids.

## Concepts

### `Array`

This is a basic 2D array type that contains various conveniences for iteration
and subarrays.

### `Grid` and `Layer`

`Grid` represents a grid layout. A grid has a content area with optional
padding. Use `new_grid_layer` to make a `LayerBuffer` whose extent is the 
unpadded content area, and `new_full_layer` to make a `LayerBuffer` that covers
the entire grid with padding.

## Examples

A basic Sudoku with no clues outside the grid could be represented as follows:

```rust
struct SudokuPuzzle {
    grid: Grid,
    digits: LayerBuffer<Option<u8>>,
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
```

This representation does not distinguish between given digits and digits entered
by a solver. This could be addressed either by adding a separate layer for
givens, or by using a cell value type other than `Option<u8>` that can
distinguish between givens and non-givens.