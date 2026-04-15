use crate::{
    array::{Array, ArrayBufferMut},
    grid::{Grid, LayerVec},
};

pub trait IteratorExt: Iterator {
    fn assign_to_array<B>(self, arr: &mut Array<B>)
    where
        Self: Sized,
        B: ArrayBufferMut<Item = Self::Item>,
    {
        arr.assign_from(self);
    }

    fn into_full_layer(self, grid: &Grid) -> LayerVec<Self::Item>
    where
        Self: Sized,
        Self::Item: Default,
    {
        let mut layer = grid.new_full_layer();
        layer.cells_mut().assign_from(self);
        layer
    }

    fn into_grid_layer(self, grid: &Grid) -> LayerVec<Self::Item>
    where
        Self: Sized,
        Self::Item: Default,
    {
        let mut layer = grid.new_grid_layer();
        layer.cells_mut().assign_from(self);
        layer
    }
}

impl<It> IteratorExt for It where It: Iterator {}
