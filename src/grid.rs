use std::rc::Rc;

use crate::array::{Array, ArrayVec, ArrayView, ArrayViewMut};

pub struct GridBuilder {
    inner: GridInner,
}

impl GridBuilder {
    pub fn new(rows: usize, cols: usize) -> GridBuilder {
        let inner = GridInner {
            full_rows: rows,
            full_cols: cols,
            grid_row0: 0,
            grid_col0: 0,
            grid_rows: rows,
            grid_cols: cols,
        };
        GridBuilder { inner }
    }

    pub fn pad(mut self, top: usize, right: usize, bottom: usize, left: usize) -> GridBuilder {
        self.inner.full_rows = self.inner.grid_rows + top + bottom;
        self.inner.full_cols = self.inner.grid_cols + left + right;
        self.inner.grid_row0 = top;
        self.inner.grid_col0 = left;
        self
    }

    pub fn build(self) -> Grid {
        Grid {
            inner: Rc::new(self.inner),
        }
    }
}

#[derive(Clone)]
pub struct Grid {
    inner: Rc<GridInner>,
}

impl Grid {
    pub fn new_full_layer<T: Default>(&self) -> LayerBuffer<T> {
        LayerBuffer::new(
            self.clone(),
            0,
            0,
            self.inner.full_rows,
            self.inner.full_cols,
        )
    }

    pub fn new_grid_layer<T: Default>(&self) -> LayerBuffer<T> {
        LayerBuffer::new(
            self.clone(),
            self.inner.grid_row0,
            self.inner.grid_col0,
            self.inner.grid_rows,
            self.inner.grid_cols,
        )
    }
}

struct GridInner {
    full_rows: usize,
    full_cols: usize,
    grid_row0: usize,
    grid_col0: usize,
    grid_rows: usize,
    grid_cols: usize,
}

#[allow(unused)]
pub struct Layer<T, B> {
    grid: Grid,
    row0: usize,
    col0: usize,
    data: Array<T, B>,
}

pub type LayerBuffer<T> = Layer<T, Vec<T>>;
pub type LayerView<'a, T> = Layer<T, &'a [T]>;

impl<T> LayerBuffer<T> {
    fn new(grid: Grid, row0: usize, col0: usize, rows: usize, cols: usize) -> LayerBuffer<T>
    where
        T: Default,
    {
        let data_rows = rows * 2 - 1;
        let data_cols = cols * 2 - 1;
        let data = ArrayVec::new_default(data_rows, data_cols);
        Layer {
            grid,
            row0,
            col0,
            data,
        }
    }

    pub fn cells<'a>(&'a self) -> ArrayView<'a, T> {
        self.data.as_ref().spaced(1, 1)
    }

    pub fn cells_mut<'a>(&'a mut self) -> ArrayViewMut<'a, T> {
        self.data.as_mut().spaced(1, 1)
    }
}
