//! This module provides the [`Array`] type for accessing a flat buffer of
//! values as a 2D array.
//!
//! Most functionality is available to buffers implementing `AsRef<[T]>` or
//! `AsMut<[T]>`. In addition to the buffer itself, [`Array`] also includes
//! parameters for mapping (row, column) pairs or linear indices to offsets into
//! the underlying buffer. Many array operations, such as transformations or
//! creating subviews, are implemented by altering these parameters and leaving
//! the underlying buffer untouched.
//!
//! # Indexing
//!
//! Array elements can be accessed either with `(row, column)` pairs, or with a
//! linear index, i.e. a row-major offset:
//!
//! ```rust
//! # use puzzle_grid::array::Array;
//! // Our buffer:
//! let squares: [usize; 6] = [0, 1, 4, 9, 16, 25];
//!
//! // Wrap the buffer as a 2x3 array:
//! //  [0,  1,  4,
//! //   9, 16, 25]
//! let arr = Array::new(2, 3, squares);
//!
//! // (row, column) access:
//! assert_eq!(arr[(1, 1)], 16);
//!
//! // row-major offset access:
//! assert_eq!(arr[4], 16);
//! ```
//!
//! Together with functions like [`len`][`Array::len()`], arrays can be iterated
//! over using for loops:
//!
//! ```
//! # use puzzle_grid::array::Array;
//! # let squares: [usize; 6] = [0, 1, 4, 9, 16, 25];
//! # let arr = Array::new(2, 3, squares);
//! for i in 0..arr.len() {
//!     assert_eq!(arr[i], i * i);
//! }
//!
//! for row in 0..arr.rows() {
//!     for col in 0..arr.cols() {
//!         let i = row * 3 + col;
//!         assert_eq!(arr[(row, col)], i * i);
//!     }
//! }
//! ```
//!
//! # Iteration
//!
//! In addition to accessing arrays with for loops as described above, arrays
//! can also be accessed using various iterators:
//!
//! ```
//! # use puzzle_grid::array::Array;
//! let squares: [usize; 6] = [0, 1, 4, 9, 16, 25];
//! let arr = Array::new(2, 3, squares);
//!
//! for (i, x) in arr.iter().enumerate() {
//!     assert_eq!(*x, i * i);
//! }
//! ```
//!
//! These iterators implement the [`ArrayIterator`] trait, which allows row and
//! column information to be accessed during iteration if desired:
//!
//! ```
//! # use puzzle_grid::array::Array;
//! # use puzzle_grid::array::ArrayIterator;
//! let times_table: [usize; 9] = [1, 2, 3, 2, 4, 6, 3, 6, 9];
//! let arr = Array::new(3, 3, times_table);
//!
//! for (row, col, x) in arr.iter().with_positions() {
//!     assert_eq!(*x, (row + 1) * (col + 1));
//! }
//! ```

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
        assert!(
            row0 < self.rows && row1 <= self.rows && col0 < self.cols && col1 <= self.cols,
            "view {row0},{col0}+{rows},{cols} out of bounds for array of shape {},{}",
            self.rows,
            self.cols
        );
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

    fn spaced(&self, rows: usize, cols: usize) -> ArrayAccess {
        let nr = rows + 1;
        let nc = cols + 1;
        ArrayAccess {
            rows: (self.rows + rows) / nr,
            cols: (self.cols + cols) / nc,
            row_stride: self.row_stride * nr as isize,
            col_stride: self.col_stride * nc as isize,
            offset: self.offset,
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

/// A wrapper around a buffer `B` containing values of type `T`.
///
/// See the [module-level docs](`crate::array`) for more information.
#[derive(Copy, Clone)]
pub struct Array<T, B> {
    access: ArrayAccess,
    buffer: B,
    _phantom: PhantomData<T>,
}

/// An array backed by a `Vec<T>`
pub type ArrayBuffer<T> = Array<T, Vec<T>>;

/// An array backed by a `Cow<'a, [T]>`
pub type ArrayCow<'a, T> = Array<T, Cow<'a, [T]>>;

/// An array backed by a `&'a [T]`
pub type ArrayView<'a, T> = Array<T, &'a [T]>;

/// An array backed by a `&'a mut [T]`
pub type ArrayViewMut<'a, T> = Array<T, &'a mut [T]>;

impl<T, B> Array<T, B> {
    /// Consumes the `Array` and returns the underlying buffer.
    pub fn unwrap(self) -> B {
        self.buffer
    }

    /// Returns the total number of elements that can be accessed by this
    /// `Array` when using linear indices (i.e. `array[i]`).
    pub fn len(&self) -> usize {
        self.access.len()
    }

    /// Returns the total number of rows that can be accessed by this `Array`.
    pub fn rows(&self) -> usize {
        self.access.rows
    }

    /// Returns the total number of columns that can be accessed by this `Array`.
    pub fn cols(&self) -> usize {
        self.access.cols
    }

    /// Returns the shape of this array as a `(rows, columns)` pair. This also
    /// represents the bounds of (row, column) accesses (i.e. `array[(r, c)]`).
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

    /// Attempts to convert this `Array` into one with a different shape. This
    /// can fail for a few different reasons:
    ///
    /// - The old and new shapes have a different total size, i.e. `rows * cols`
    ///   is different before and after.
    ///
    /// - The current access pattern is not contiguous and strictly increasing,
    ///   e.g. this `Array` represents a subview or a transformation. An array
    ///   can be made contiguous with the
    ///   [`as_contiguous`](`Self::as_contiguous`) method, but this may require
    ///   cloning the data.
    pub fn reshape(self, rows: usize, cols: usize) -> Option<Self> {
        self.try_map_access(|a| a.reshape(rows, cols))
    }

    /// Returns an `Array` that accesses values by skipping `rows` and `cols`
    /// For example, using `spaced(1, 1)` on a 9x9 `Array` would return an
    /// 5x5 `Array` that accesses values at odd (1-indexed) rows and columns.
    pub fn spaced(self, rows: usize, cols: usize) -> Self {
        self.map_access(|a| a.spaced(rows, cols))
    }

    /// Returns the transpose of this `Array`, i.e. the array flipped diagonally
    /// so that rows become columns and vice versa.
    pub fn transpose(self) -> Self {
        self.map_access(ArrayAccess::transpose)
    }

    /// Returns this `Array` flipped horizontally.
    pub fn flip_h(self) -> Self {
        self.map_access(ArrayAccess::flip_h)
    }

    /// Returns this `Array` flipped vertically.
    pub fn flip_v(self) -> Self {
        self.map_access(ArrayAccess::flip_v)
    }

    /// Returns this `Array` rotated 90 degrees clockwise.
    pub fn rotate_cw(self) -> Self {
        self.map_access(ArrayAccess::rotate_cw)
    }

    /// Returns this `Array` rotated 180 degrees.
    pub fn rotate_180(self) -> Self {
        self.map_access(ArrayAccess::rotate_180)
    }

    /// Returns this `Array` rotated 90 degrees counter-clockwise.
    pub fn rotate_ccw(self) -> Self {
        self.map_access(ArrayAccess::rotate_ccw)
    }
}

impl<T> ArrayBuffer<T> {
    /// Creates a new array of shape `(rows, cols)` initialized with
    /// `T::default()` and backed by a `Vec<T>`.
    pub fn new_default(rows: usize, cols: usize) -> ArrayBuffer<T>
    where
        T: Default,
    {
        (0..rows * cols)
            .map(|_| T::default())
            .collect::<Self>()
            .reshape(rows, cols)
            .unwrap()
    }
}

impl<'a, T: Clone> ArrayCow<'a, T> {
    /// Converts this `Array` into one backed by a `Vec<T>`. If the underlying
    /// buffer is borrowed, the data will be cloned.
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

impl<'a, T> ArrayView<'a, T> {
    /// Converts this `Array` into a view of the given row.
    ///
    /// See [`Array::row`] for more information.
    pub fn into_row(self, row: usize) -> ArrayView<'a, T> {
        let cols = self.cols();
        self.into_view(row, 0, 1, cols)
    }

    /// Converts this `Array` into a view of the given column.
    ///
    /// See [`Array::col`] for more information.
    pub fn into_col(self, col: usize) -> ArrayView<'a, T> {
        let rows = self.rows();
        self.into_view(0, col, rows, 1)
    }

    /// Converts this `Array` into a view of the given region.
    ///
    /// See [`Array::view`] for more information.
    pub fn into_view(self, row0: usize, col0: usize, rows: usize, cols: usize) -> ArrayView<'a, T> {
        Array {
            access: self.access.view(row0, col0, rows, cols),
            buffer: self.buffer,
            _phantom: PhantomData,
        }
    }
}

impl<T, B> Array<T, B>
where
    B: AsRef<[T]>,
{
    /// Creates a new array of shape `(rows, cols)` that wraps the given buffer.
    ///
    /// # Panics
    ///
    /// This function panics if the buffer size is not `rows * cols`.
    pub fn new(rows: usize, cols: usize, buf: B) -> Array<T, B> {
        let access = ArrayAccess::new(rows, cols);
        assert!(
            buf.as_ref().len() == access.len(),
            "cannot create array of shape {rows},{cols} with buffer of length {}",
            buf.as_ref().len()
        );
        Array {
            access,
            buffer: buf,
            _phantom: PhantomData,
        }
    }

    /// Creates a new array of shape `(1, buf.len())` that wraps the given buffer.
    pub fn flat(buf: B) -> Array<T, B> {
        Array::new(1, buf.as_ref().len(), buf)
    }

    /// If this `Array` represents contiguous and strictly increasing access of
    /// the underlying buffer, returns an array with a `Cow::Borrowed` buffer.
    /// Otherwise, the accessed data is cloned and a `Cow::Owned` is returned.
    ///
    /// This method is useful for guaranteeing that a [`reshape`][`Self::reshape`]
    /// will succeed:
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// let buf: [u8; 4] = [1, 2, 3, 4];
    /// let arr = Array::new(2, 2, buf);
    ///
    /// // create a view of the first column
    /// let col = arr.col(0);
    /// assert_eq!(col.shape(), (2, 1));
    ///
    /// // this fails because col accesses buf[0] and buf[2], which is not contiguous
    /// assert!(col.reshape(1, 2).is_none());
    ///
    /// // however we can make it contiguous and the reshape succeeds:
    /// assert!(col.as_contiguous().reshape(1, 2).is_some());
    /// ```
    ///
    /// Note that [`reshape`][`Self::reshape`] can still fail if the old and new
    /// sizes do not match, so be careful when using `.unwrap()`.
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

    /// Creates an `Array` backed by a reference to this array's buffer.
    pub fn as_ref<'a>(&'a self) -> ArrayView<'a, T> {
        Array {
            access: self.access,
            buffer: self.buffer.as_ref(),
            _phantom: PhantomData,
        }
    }

    /// Creates a view of a given row of the array.
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// let arr = Array::new(2, 2, [1, 2, 3, 4]);
    /// let row = arr.row(1);
    ///
    /// assert_eq!(row[0], 3);
    /// assert_eq!(row[1], 4);
    /// ```
    pub fn row<'a>(&'a self, row: usize) -> ArrayView<'a, T> {
        self.view(row, 0, 1, self.cols())
    }

    /// Creates a view of a given column of the array.
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// let arr = Array::new(2, 2, [1, 2, 3, 4]);
    /// let col = arr.col(1);
    ///
    /// assert_eq!(col[0], 2);
    /// assert_eq!(col[1], 4);
    /// ```
    pub fn col<'a>(&'a self, col: usize) -> ArrayView<'a, T> {
        self.view(0, col, self.rows(), 1)
    }

    /// Creates a view of a given region of the array.
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// # use puzzle_grid::array::ArrayBuffer;
    /// // Create the following array:
    /// //  1  2  3
    /// //  4  5  6
    /// //  7  8  9
    /// let arr = ArrayBuffer::new(3, 3, (1..=9).collect());
    ///
    /// // Create a view of the 2x2 in the bottom right:
    /// let view = arr.view(1, 1, 2, 2);
    ///
    /// assert_eq!(view[(0, 0)], 5);
    /// assert_eq!(view[(0, 1)], 6);
    /// assert_eq!(view[(1, 0)], 8);
    /// assert_eq!(view[(1, 1)], 9);
    /// ```
    pub fn view<'a>(
        &'a self,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
    ) -> ArrayView<'a, T> {
        Array {
            access: self.access.view(row0, col0, rows, cols),
            buffer: self.buffer.as_ref(),
            _phantom: PhantomData,
        }
    }

    /// Creates an iterator that visits all elements of the array in row-major
    /// (i.e. linear) order.
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// let arr = Array::new(2, 2, [1, 2, 3, 4]);
    /// let mut it = arr.iter();
    ///
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&3));
    /// assert_eq!(it.next(), Some(&4));
    /// assert_eq!(it.next(), None);
    /// ```
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter {
            access: self.access,
            buffer: self.buffer.as_ref(),
            next: 0,
        }
    }

    /// Creates an iterator over all valid views of shape `(rows, cols)`. For
    /// example, `arr.iter_views(2, 2)` is an iterator that visits all valid 2x2
    /// views of `arr`.
    ///
    /// # Panics
    ///
    /// This function panics if `rows` or `cols` is zero, since the meaning of
    /// iteration over empty views is unclear.
    pub fn iter_views<'a>(&'a self, rows: usize, cols: usize) -> IterViews<'a, T> {
        if rows == 0 || cols == 0 {
            panic!("iter_views({rows}, {cols}): must be nonempty");
        }
        IterViews {
            access: self.access,
            buffer: self.buffer.as_ref(),
            next: 0,
            rows,
            cols,
        }
    }

    /// Creates an iterator over all rows of the array. The returned views will
    /// have shape `(1, self.cols())`.
    ///
    /// # Panics
    ///
    /// This function panics if `self.cols()` is 0.
    pub fn iter_rows<'a>(&'a self) -> IterViews<'a, T> {
        self.iter_views(1, self.cols())
    }

    /// Creates an iterator over all columns of the array. The returned views
    /// will have shape `(self.rows(), 1)`.
    ///
    /// # Panics
    ///
    /// This function panics if `self.rows()` is 0.
    pub fn iter_cols<'a>(&'a self) -> IterViews<'a, T> {
        self.iter_views(self.rows(), 1)
    }
}

impl<T, B> Array<T, B>
where
    B: AsMut<[T]>,
{
    /// Creates an `Array` backed by a mutable reference to this array's buffer.
    pub fn as_mut<'a>(&'a mut self) -> ArrayViewMut<'a, T> {
        Array {
            access: self.access,
            buffer: self.buffer.as_mut(),
            _phantom: PhantomData,
        }
    }

    /// Creates a mutable view of the given row.
    pub fn row_mut<'a>(&'a mut self, row: usize) -> ArrayViewMut<'a, T> {
        self.view_mut(row, 0, 1, self.cols())
    }

    /// Creates a mutable view of the given column.
    pub fn col_mut<'a>(&'a mut self, col: usize) -> ArrayViewMut<'a, T> {
        self.view_mut(0, col, self.rows(), 1)
    }

    /// Creates a mutable view of the given region.
    pub fn view_mut<'a>(
        &'a mut self,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
    ) -> ArrayViewMut<'a, T> {
        Array {
            access: self.access.view(row0, col0, rows, cols),
            buffer: self.buffer.as_mut(),
            _phantom: PhantomData,
        }
    }

    /// Creates an iterator that visits all elements of the array in row-major
    /// (i.e. linear) order and returns a `&mut` reference to them.
    ///
    /// # Safety
    ///
    /// The returned iterator is unsafe to use if the array's access pattern
    /// is aliasing, i.e. if there are distinct `(row, col)` pairs that map to
    /// the same index in the underlying buffer. It's not currently possible
    /// to achieve this using only safe code, however.
    ///
    /// In the future, this function will panic if the access pattern is
    /// aliasing, but this check is not currently implemented.
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        IterMut {
            access: self.access,
            buffer: self.buffer.as_mut(),
            next: 0,
        }
    }

    /// Assign all values from the given iterator to this array, starting at
    /// index zero. This function *does not* check if the iterators are the
    /// same length. Assignment stops whenever either iterator runs out.
    pub fn assign_from<It>(&mut self, other: It)
    where
        It: Iterator<Item = T>,
    {
        self.iter_mut().zip(other).for_each(|(x, y)| *x = y);
    }
}

impl<T, B> Index<usize> for Array<T, B>
where
    B: AsRef<[T]>,
{
    type Output = T;

    /// Index into the array using a row-major offset.
    fn index(&self, index: usize) -> &Self::Output {
        let row = index / self.access.cols;
        let col = index % self.access.cols;
        &self[(row, col)]
    }
}

impl<T, B> IndexMut<usize> for Array<T, B>
where
    B: AsRef<[T]> + AsMut<[T]>,
{
    /// Index into the array using a row-major offset.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = index / self.access.cols;
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
        assert!(
            row < self.rows() && col < self.cols(),
            "index ({row},{col}) out of bounds for array of shape {},{}",
            self.rows(),
            self.cols()
        );
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
        assert!(
            row < self.rows() && col < self.cols(),
            "index ({row},{col}) out of bounds for array of shape {},{}",
            self.rows(),
            self.cols()
        );
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

/// A trait for iterators over [`Array`]s.
pub trait ArrayIterator: Iterator {
    /// Returns the next item in this iterator, together with its row and column.
    ///
    /// ```rust
    /// # use puzzle_grid::array::Array;
    /// # use puzzle_grid::array::ArrayIterator;
    /// let arr = Array::new(2, 2, [1, 2, 3, 4]);
    /// let mut it = arr.iter();
    ///
    /// assert_eq!(it.next_with_position(), Some((0, 0, &1)));
    /// assert_eq!(it.next_with_position(), Some((0, 1, &2)));
    /// assert_eq!(it.next_with_position(), Some((1, 0, &3)));
    /// assert_eq!(it.next_with_position(), Some((1, 1, &4)));
    /// assert_eq!(it.next_with_position(), None);
    /// ```
    fn next_with_position(&mut self) -> Option<(usize, usize, Self::Item)>;

    /// Returns an iterator over `(row, column, item)` triples, similar to
    /// [`Iterator::enumerate`]. The iterator just calls
    /// [`next_with_position`][`Self::next_with_position`].
    fn with_positions(self) -> WithPositions<Self>
    where
        Self: Sized,
    {
        WithPositions { inner: self }
    }
}

/// An iterator over `(row, column, item)` triples.
///
/// Returned by [`ArrayIterator::with_positions`].
pub struct WithPositions<It> {
    inner: It,
}

impl<It: ArrayIterator> Iterator for WithPositions<It> {
    type Item = (usize, usize, It::Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_with_position()
    }
}

/// An iterator over references to items of the array, visited in row-major order.
///
/// Returned by [`Array::iter`].
pub struct Iter<'a, T> {
    access: ArrayAccess,
    buffer: &'a [T],
    next: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_with_position().map(|(_, _, x)| x)
    }
}

impl<'a, T> ArrayIterator for Iter<'a, T> {
    fn next_with_position(&mut self) -> Option<(usize, usize, Self::Item)> {
        if self.next >= self.access.len() {
            return None;
        }

        let index = self.next;
        self.next += 1;

        let row = index / self.access.cols;
        let col = index % self.access.cols;
        let idx = self.access.to_offset(row as isize, col as isize);

        Some((row, col, &self.buffer[idx as usize]))
    }
}

/// An iterator over mutable references to items of the array, visited in
/// row-major order.
///
/// Returned by [`Array::iter_mut`].
pub struct IterMut<'a, T> {
    access: ArrayAccess,
    buffer: &'a mut [T],
    next: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_with_position().map(|(_, _, x)| x)
    }
}

impl<'a, T> ArrayIterator for IterMut<'a, T> {
    fn next_with_position(&mut self) -> Option<(usize, usize, Self::Item)> {
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
            self.buffer
                .as_mut_ptr()
                .add(idx as usize)
                .as_mut()
                .map(|x| (row, col, x))
        }
    }
}

/// An iterator over array views of a given size.
///
/// See [`Array::iter_views`] for more information.
pub struct IterViews<'a, T> {
    access: ArrayAccess,
    buffer: &'a [T],
    next: usize,
    rows: usize,
    cols: usize,
}

impl<'a, T> Iterator for IterViews<'a, T> {
    type Item = Array<T, &'a [T]>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_with_position().map(|(_, _, x)| x)
    }
}

impl<'a, T> ArrayIterator for IterViews<'a, T> {
    fn next_with_position(&mut self) -> Option<(usize, usize, Self::Item)> {
        if self.rows > self.access.rows || self.cols > self.access.cols {
            return None;
        }

        let itrows = self.access.rows - self.rows + 1;
        let itcols = self.access.cols - self.cols + 1;
        if self.next >= itrows * itcols {
            return None;
        }

        let index = self.next;
        self.next += 1;

        let row = index / itcols;
        let col = index % itcols;

        let item = Array {
            access: self.access.view(row, col, self.rows, self.cols),
            buffer: self.buffer,
            _phantom: PhantomData,
        };
        Some((row, col, item))
    }
}
