use std::{
    borrow::Cow,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

// NOTE: it is critical for correctness that aliasing ArrayAccesses, i.e. those
// that map distinct (row, col) accesses to the same buffer index, are never
// created.

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

    /// Returns whether iterating over this ArrayAccess in row-major order
    /// visits a contiguous chunk of the buffer at increasing indices.
    fn is_contiguous_increasing(&self) -> bool {
        self.col_stride == 1 && (self.row_stride == self.cols as isize || self.rows <= 1)
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

    fn reshape(&self, rows: usize, cols: usize) -> Option<ArrayAccess> {
        if self.rows * self.cols != rows * cols {
            return None;
        }
        if self.is_contiguous_increasing() {
            Some(ArrayAccess {
                rows,
                cols,
                row_stride: cols as isize,
                col_stride: 1,
                offset: self.offset,
            })
        } else {
            None
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

pub type ArrayBuffer<T> = Array<T, Vec<T>>;
pub type ArrayCow<'a, T> = Array<T, Cow<'a, [T]>>;
pub type ArrayView<'a, T> = Array<T, &'a [T]>;
pub type ArrayViewMut<'a, T> = Array<T, &'a mut [T]>;

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

    fn try_map_access<F>(self, f: F) -> Option<Self>
    where
        F: FnOnce(&ArrayAccess) -> Option<ArrayAccess>,
    {
        Some(Array {
            access: f(&self.access)?,
            buffer: self.buffer,
            _phantom: PhantomData,
        })
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

    pub fn reshape(self, rows: usize, cols: usize) -> Option<Self> {
        self.try_map_access(|a| a.reshape(rows, cols))
    }
}

impl<'a, T: Clone> ArrayCow<'a, T> {
    pub fn into_owned(self) -> ArrayBuffer<T> {
        match self.buffer {
            Cow::Borrowed(_) => self.iter().cloned().collect(),
            Cow::Owned(buffer) => Array {
                access: self.access,
                buffer,
                _phantom: PhantomData,
            },
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

    pub fn flat(buf: B) -> Array<T, B> {
        Array::new(1, buf.as_ref().len(), buf)
    }

    pub fn as_contiguous<'a>(&'a self) -> ArrayCow<'a, T>
    where
        T: Clone,
    {
        if self.access.is_contiguous_increasing() {
            Array {
                access: self.access,
                buffer: Cow::Borrowed(self.buffer.as_ref()),
                _phantom: PhantomData,
            }
        } else {
            let buffer = self.iter().cloned().collect();
            Array {
                access: ArrayAccess::new(self.rows(), self.cols()),
                buffer: Cow::Owned(buffer),
                _phantom: PhantomData,
            }
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

    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter {
            access: self.access,
            buffer: self.buffer.as_ref(),
            next: 0,
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

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        IterMut {
            access: self.access,
            buffer: self.buffer.as_mut(),
            next: 0,
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

impl<T, B> FromIterator<T> for Array<T, B>
where
    B: FromIterator<T> + AsRef<[T]>,
{
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        Array::flat(B::from_iter(iter))
    }
}

pub struct Iter<'a, T> {
    access: ArrayAccess,
    buffer: &'a [T],
    next: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.access.len() {
            return None;
        }

        let index = self.next;
        self.next += 1;

        let row = index / self.access.cols;
        let col = index % self.access.cols;
        let idx = self.access.to_offset(row as isize, col as isize);

        Some(&self.buffer[idx as usize])
    }
}

pub struct IterMut<'a, T> {
    access: ArrayAccess,
    buffer: &'a mut [T],
    next: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.access.len() {
            return None;
        }

        let index = self.next;
        self.next += 1;

        let row = index / self.access.cols;
        let col = index % self.access.cols;
        let idx = self.access.to_offset(row as isize, col as isize);

        // SAFETY: ArrayAccess can never be aliasing
        unsafe {
            assert!(0 <= idx && (idx as usize) < self.buffer.len());
            self.buffer.as_mut_ptr().add(idx as usize).as_mut()
        }
    }
}
