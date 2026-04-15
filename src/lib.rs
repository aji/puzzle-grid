//! # Overview
//!
//! This crate provides useful data structures for logic puzzle grids. They are
//! generic over the grid contents.
//!
//! # Array API ([`mod@array`])
//!
//! The [`mod@array`] module contains an [`Array`](`array::Array`) type for
//! accessing a flat buffer of values as a 2D array. Transformed views and
//! subarray views can be created with methods like
//! [`flip_h`](`array::Array::flip_h`) and [`view`](`array::Array::view`).
//! Various iteration patterns are provided as well, such as
//! [`iter_cols`](`array::Array::iter_cols`) to iterate over views of the
//! array columns.
//!
//! # Grid API ([`grid`])
//!
//! The [`Grid`](`grid::Grid`) type represents a puzzle grid layout, with a main
//! grid area and optional padding area. [`Layer`](`grid::Layer`)s can be
//! constructed either in the grid area itself or the full area with padding.
//! A [`Layer`](`grid::Layer`) stores cell, edge, and corner values in a single
//! [`Array`](`array::Array`) and provides methods for accessing these
//! subarrays.

pub mod array;
pub mod grid;
pub mod iter;
