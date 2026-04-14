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
    offset: usize,
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

    fn to_offset(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows && col < self.cols);
        self.offset + (self.row_stride * row as isize + self.col_stride * col as isize) as usize
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
            offset: self.to_offset(row0, col0),
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
}

#[derive(Clone)]
pub struct Array<T, B> {
    access: ArrayAccess,
    buffer: B,
    _phantom: PhantomData<T>,
}

impl<T, B> Array<T, B> {
    pub fn len(&self) -> usize {
        self.access.len()
    }

    pub fn transpose(self) -> Self {
        Array {
            access: self.access.transpose(),
            buffer: self.buffer,
            _phantom: PhantomData,
        }
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
        let idx = self.access.to_offset(row, col);
        let buf = self.buffer.as_ref();
        &buf[idx]
    }
}

impl<T, B> IndexMut<(usize, usize)> for Array<T, B>
where
    B: AsRef<[T]> + AsMut<[T]>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let idx = self.access.to_offset(row, col);
        let buf = self.buffer.as_mut();
        &mut buf[idx]
    }
}
