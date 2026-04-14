use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

#[derive(Copy, Clone)]
struct ArrayAccess {
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
    offset: isize,
}

impl ArrayAccess {
    const fn new(rows: usize, cols: usize) -> ArrayAccess {
        ArrayAccess {
            rows,
            cols,
            row_stride: cols as isize,
            col_stride: 1,
            offset: 0,
        }
    }

    const fn len(&self) -> usize {
        self.rows * self.cols
    }

    fn to_offset(&self, row: isize, col: isize) -> isize {
        let base_offset = self.offset as isize;
        let row_offset = self.row_stride * row;
        let col_offset = self.col_stride * col;
        base_offset + row_offset + col_offset
    }

    fn view(&self, row0: usize, col0: usize, rows: usize, cols: usize) -> ArrayAccess {
        let row1 = row0 + rows;
        let col1 = col0 + cols;
        assert!(row0 < self.rows && row1 <= self.rows);
        assert!(col0 < self.cols && col1 <= self.cols);
        ArrayAccess {
            rows,
            cols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            offset: self.to_offset(row0 as isize, col0 as isize),
        }
    }

    fn transpose(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            offset: self.offset,
        }
    }

    fn flip_h(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.rows,
            cols: self.cols,
            row_stride: self.row_stride,
            col_stride: -self.col_stride,
            offset: self.to_offset(0, self.cols as isize - 1),
        }
    }

    fn flip_v(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.rows,
            cols: self.cols,
            row_stride: -self.row_stride,
            col_stride: self.col_stride,
            offset: self.to_offset(self.rows as isize - 1, 0),
        }
    }

    fn rotate_cw(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: -self.row_stride,
            offset: self.to_offset(self.rows as isize - 1, 0),
        }
    }

    fn rotate_180(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.rows,
            cols: self.cols,
            row_stride: -self.row_stride,
            col_stride: -self.col_stride,
            offset: self.to_offset(self.rows as isize - 1, self.cols as isize - 1),
        }
    }

    fn rotate_ccw(&self) -> ArrayAccess {
        ArrayAccess {
            rows: self.cols,
            cols: self.rows,
            row_stride: -self.col_stride,
            col_stride: self.row_stride,
            offset: self.to_offset(0, self.cols as isize - 1),
        }
    }
}

#[derive(Copy, Clone)]
pub struct Array<T, B> {
    access: ArrayAccess,
    buffer: B,
    _phantom: PhantomData<T>,
}

impl<T, B> Array<T, B> {
    pub fn unwrap(self) -> B {
        self.buffer
    }

    pub fn len(&self) -> usize {
        self.access.len()
    }

    pub fn rows(&self) -> usize {
        self.access.rows
    }

    pub fn cols(&self) -> usize {
        self.access.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn map_access<F>(self, f: F) -> Self
    where
        F: FnOnce(&ArrayAccess) -> ArrayAccess,
    {
        Array {
            access: f(&self.access),
            buffer: self.buffer,
            _phantom: PhantomData,
        }
    }

    pub fn transpose(self) -> Self {
        self.map_access(ArrayAccess::transpose)
    }

    pub fn flip_h(self) -> Self {
        self.map_access(ArrayAccess::flip_h)
    }

    pub fn flip_v(self) -> Self {
        self.map_access(ArrayAccess::flip_v)
    }

    pub fn rotate_cw(self) -> Self {
        self.map_access(ArrayAccess::rotate_cw)
    }

    pub fn rotate_180(self) -> Self {
        self.map_access(ArrayAccess::rotate_180)
    }

    pub fn rotate_ccw(self) -> Self {
        self.map_access(ArrayAccess::rotate_ccw)
    }
}

impl<T, B> Array<T, B>
where
    B: AsRef<[T]>,
{
    pub fn new(rows: usize, cols: usize, buf: B) -> Array<T, B> {
        let access = ArrayAccess::new(rows, cols);
        assert!(buf.as_ref().len() == access.len());
        Array {
            access,
            buffer: buf,
            _phantom: PhantomData,
        }
    }

    pub fn as_ref<'a>(&'a self) -> Array<T, &'a [T]> {
        Array {
            access: self.access,
            buffer: self.buffer.as_ref(),
            _phantom: PhantomData,
        }
    }

    pub fn view<'a>(
        &'a self,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
    ) -> Array<T, &'a [T]> {
        Array {
            access: self.access.view(row0, col0, rows, cols),
            buffer: self.buffer.as_ref(),
            _phantom: PhantomData,
        }
    }
}

impl<T, B> Array<T, B>
where
    B: AsMut<[T]>,
{
    pub fn as_mut<'a>(&'a mut self) -> Array<T, &'a mut [T]> {
        Array {
            access: self.access,
            buffer: self.buffer.as_mut(),
            _phantom: PhantomData,
        }
    }

    pub fn view_mut<'a>(
        &'a mut self,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
    ) -> Array<T, &'a mut [T]> {
        Array {
            access: self.access.view(row0, col0, rows, cols),
            buffer: self.buffer.as_mut(),
            _phantom: PhantomData,
        }
    }
}

impl<T, B> Index<usize> for Array<T, B>
where
    B: AsRef<[T]>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let row = index / self.access.rows;
        let col = index % self.access.cols;
        &self[(row, col)]
    }
}

impl<T, B> IndexMut<usize> for Array<T, B>
where
    B: AsRef<[T]> + AsMut<[T]>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = index / self.access.rows;
        let col = index % self.access.cols;
        &mut self[(row, col)]
    }
}

impl<T, B> Index<(usize, usize)> for Array<T, B>
where
    B: AsRef<[T]>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows() && col < self.cols());
        let idx = self.access.to_offset(row as isize, col as isize);
        let buf = self.buffer.as_ref();
        assert!(0 <= idx && (idx as usize) < buf.len());
        &buf[idx as usize]
    }
}

impl<T, B> IndexMut<(usize, usize)> for Array<T, B>
where
    B: AsRef<[T]> + AsMut<[T]>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows() && col < self.cols());
        let idx = self.access.to_offset(row as isize, col as isize);
        let buf = self.buffer.as_mut();
        assert!(0 <= idx && (idx as usize) < buf.len());
        &mut buf[idx as usize]
    }
}
