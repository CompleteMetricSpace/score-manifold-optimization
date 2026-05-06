#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base and affine manifold abstractions."""

import math
from abc import abstractmethod
from typing import Optional, Tuple

import torch

from ..base import Region, Space
from ..euclidean import EuclideanSpace, TrajectorySpace, VectorSpace


class Manifold(Region):
    """
    Smooth submanifold M ⊂ Space (extends Region).

    A manifold has additional Riemannian structure:
    - Tangent spaces T_x M at each point x
    - Normal spaces N_x M orthogonal to tangent
    - Projection onto tangent space
    - Riemannian geometry operations

    Note: Manifolds are used for evaluation/metrics only.
    They do NOT influence training or sampling in the diffusion process.
    """

    @abstractmethod
    def get_dim_manifold(self) -> int:
        """
        Return intrinsic dimension of the manifold.

        Returns:
            Integer dimension of the manifold
        """
        pass

    @abstractmethod
    def get_dim_ambient(self) -> int:
        """
        Return dimension of ambient space.

        Returns:
            Integer dimension of ambient Euclidean space
        """
        pass

    @abstractmethod
    def get_normal_tangent_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get orthonormal bases for normal and tangent spaces at x.

        Args:
            x: (batch, *space.dims) points on manifold

        Returns:
            N: (batch, k_normal, *space.dims) - normal basis vectors
            T: (batch, k_tangent, *space.dims) - tangent basis vectors
        """
        pass

    @abstractmethod
    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto tangent space T_x M.

        Args:
            x: (batch, *space.dims) points on manifold
            v: (batch, *space.dims) vectors in ambient space

        Returns:
            v_tangent: (batch, *space.dims) projection onto T_x M
        """
        pass


class AffineSubspace(Manifold):
    """
    Affine subspace defined by linear equalities A vec(x) = b.

    Supports arbitrary ambient tensor shapes by flattening points internally.
    Optionally binds to an explicit Space (including TrajectorySpace).
    """

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ambient_shape: Tuple[int, ...] = None,
        space: Optional[Space] = None,
        rank_tol: float = 1e-7,
        consistency_tol: float = 1e-6,
        sample_std: float = 1.0,
    ):
        A = torch.as_tensor(A)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D with shape (m, D), got shape {tuple(A.shape)}")
        if not A.is_floating_point():
            A = A.to(torch.float32)

        b = torch.as_tensor(b, device=A.device, dtype=A.dtype)
        if b.ndim == 2 and b.shape[1] == 1:
            b = b.squeeze(1)
        if b.ndim != 1:
            raise ValueError(f"b must be 1D with shape (m,), got shape {tuple(b.shape)}")
        if b.shape[0] != A.shape[0]:
            raise ValueError(f"Incompatible A, b shapes: {tuple(A.shape)} and {tuple(b.shape)}")

        if rank_tol < 0:
            raise ValueError("rank_tol must be non-negative")
        if consistency_tol < 0:
            raise ValueError("consistency_tol must be non-negative")
        if sample_std < 0:
            raise ValueError("sample_std must be non-negative")

        self.rank_tol = float(rank_tol)
        self.consistency_tol = float(consistency_tol)
        self.sample_std = float(sample_std)

        D = int(A.shape[1])
        inferred_shape = (D,)
        if ambient_shape is not None:
            inferred_shape = tuple(int(d) for d in ambient_shape)
            if len(inferred_shape) == 0 or any(d <= 0 for d in inferred_shape):
                raise ValueError(f"ambient_shape must be non-empty positive ints, got {ambient_shape}")
            if math.prod(inferred_shape) != D:
                raise ValueError(
                    f"ambient_shape product {math.prod(inferred_shape)} does not match A.shape[1]={D}"
                )

        if space is not None:
            if not isinstance(space, Space):
                raise TypeError(f"space must be a Space instance, got {type(space)}")
            space_shape = tuple(int(d) for d in space.get_dims())
            if math.prod(space_shape) != D:
                raise ValueError(
                    f"space total dim {math.prod(space_shape)} does not match A.shape[1]={D}"
                )
            if ambient_shape is not None and tuple(inferred_shape) != tuple(space_shape):
                raise ValueError(
                    f"ambient_shape {tuple(inferred_shape)} does not match space dims {tuple(space_shape)}"
                )
            super().__init__(space)
            self.ambient_shape = tuple(space_shape)
        else:
            if ambient_shape is None:
                ambient_space: Space = VectorSpace(D)
            else:
                ambient_space = VectorSpace(D) if len(inferred_shape) == 1 else EuclideanSpace(inferred_shape)
            super().__init__(ambient_space)
            self.ambient_shape = tuple(self.get_dims())

        self.ambient_dim = int(math.prod(self.ambient_shape))
        if self.ambient_dim != D:
            raise ValueError(f"Internal dimension mismatch: {self.ambient_dim} != {D}")

        # Full SVD gives a complete orthonormal basis in Vh for both row-space and nullspace.
        _, S, Vh = torch.linalg.svd(A, full_matrices=True)
        if S.numel() == 0:
            rank = 0
        else:
            smax = float(S.max().item())
            rank = 0 if smax == 0.0 else int((S > self.rank_tol * smax).sum().item())

        self.rank = rank
        self.manifold_dim = self.ambient_dim - rank
        self.codim = rank

        self.N_flat = Vh[:rank, :].contiguous()
        self.T_flat = Vh[rank:, :].contiguous()

        x0_flat = torch.linalg.pinv(A, rtol=self.rank_tol) @ b
        residual = torch.linalg.norm(A @ x0_flat - b)
        if float(residual.item()) > self.consistency_tol:
            raise ValueError(
                f"Inconsistent affine system: ||A x0 - b||={float(residual.item()):.3e} "
                f"> consistency_tol={self.consistency_tol:.3e}"
            )

        self.x0_flat = x0_flat.contiguous()

    @classmethod
    def from_offset_basis(
        cls,
        offset: torch.Tensor,
        tangent_basis: torch.Tensor,
        *,
        space: Optional[Space] = None,
        rank_tol: float = 1e-7,
        orthonormalize: bool = True,
        sample_std: float = 1.0,
    ) -> "AffineSubspace":
        offset = torch.as_tensor(offset)
        if not offset.is_floating_point():
            offset = offset.to(torch.float32)

        offset_shape = tuple(offset.shape)
        if len(offset_shape) == 0:
            raise ValueError("offset must be at least 1D")

        D = int(math.prod(offset_shape))
        x0_flat = offset.reshape(-1)

        Tb = torch.as_tensor(tangent_basis, device=offset.device, dtype=offset.dtype)
        if Tb.ndim != len(offset_shape) + 1:
            raise ValueError(
                f"tangent_basis must have shape (k, *offset.shape), got {tuple(Tb.shape)}"
            )
        if tuple(Tb.shape[1:]) != offset_shape:
            raise ValueError(
                f"tangent_basis shape {tuple(Tb.shape)} incompatible with offset shape {offset_shape}"
            )

        Tb_flat = Tb.reshape(Tb.shape[0], D)
        if orthonormalize:
            if Tb_flat.shape[0] == 0:
                T_flat = Tb_flat
            else:
                _, S, Vh = torch.linalg.svd(Tb_flat, full_matrices=False)
                if S.numel() == 0:
                    k = 0
                else:
                    smax = float(S.max().item())
                    k = 0 if smax == 0.0 else int((S > rank_tol * smax).sum().item())
                T_flat = Vh[:k, :]
        else:
            G = Tb_flat @ Tb_flat.transpose(-1, -2)
            eye = torch.eye(Tb_flat.shape[0], device=Tb_flat.device, dtype=Tb_flat.dtype)
            if not torch.allclose(G, eye, atol=1e-5, rtol=1e-5):
                raise ValueError("tangent_basis rows must be orthonormal when orthonormalize=False")
            T_flat = Tb_flat

        k = int(T_flat.shape[0])
        if k > D:
            raise ValueError(f"tangent_basis rank {k} cannot exceed ambient dim {D}")

        if k == D:
            N_flat = torch.empty((0, D), device=T_flat.device, dtype=T_flat.dtype)
        elif k == 0:
            N_flat = torch.eye(D, device=T_flat.device, dtype=T_flat.dtype)
        else:
            Q, _ = torch.linalg.qr(T_flat.transpose(-1, -2), mode="complete")
            N_flat = Q[:, k:].transpose(-1, -2).contiguous()

        A = N_flat
        b = A @ x0_flat
        return cls(
            A=A,
            b=b,
            ambient_shape=offset_shape,
            space=space,
            rank_tol=rank_tol,
            consistency_tol=1e-6,
            sample_std=sample_std,
        )

    @classmethod
    def from_trajectory_space(
        cls,
        space: TrajectorySpace,
        A: torch.Tensor,
        b: torch.Tensor,
        rank_tol: float = 1e-7,
        consistency_tol: float = 1e-6,
        sample_std: float = 1.0,
    ) -> "AffineSubspace":
        if not isinstance(space, TrajectorySpace):
            raise TypeError(f"space must be TrajectorySpace, got {type(space)}")
        A = torch.as_tensor(A)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D with shape (m, D), got shape {tuple(A.shape)}")
        if A.shape[1] != space.get_total_dim():
            raise ValueError(
                f"A.shape[1]={A.shape[1]} does not match trajectory total dim {space.get_total_dim()}"
            )
        return cls(
            A=A,
            b=b,
            space=space,
            rank_tol=rank_tol,
            consistency_tol=consistency_tol,
            sample_std=sample_std,
        )

    def _validate_input_shape(self, x: torch.Tensor, name: str = "x"):
        if x.ndim < 2:
            raise ValueError(f"{name} must have shape (batch, *dims), got {tuple(x.shape)}")
        if tuple(x.shape[1:]) != tuple(self.ambient_shape):
            raise ValueError(
                f"{name} has shape {tuple(x.shape[1:])}, expected {tuple(self.ambient_shape)}"
            )

    def _as_flat_batch(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.ambient_dim)

    def _x0_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.x0_flat.to(device=ref.device, dtype=ref.dtype)

    def _T_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.T_flat.to(device=ref.device, dtype=ref.dtype)

    def _N_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.N_flat.to(device=ref.device, dtype=ref.dtype)

    def get_dim_manifold(self) -> int:
        return self.manifold_dim

    def get_dim_ambient(self) -> int:
        return self.ambient_dim

    def project(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x, name="x")
        x_flat = self._as_flat_batch(x)
        x0 = self._x0_on(x_flat)
        T = self._T_on(x_flat)
        coeff = (x_flat - x0) @ T.transpose(-1, -2)
        x_proj = x0.unsqueeze(0) + coeff @ T
        return x_proj.reshape(x.shape[0], *self.ambient_shape)

    def sample(self, n: int) -> torch.Tensor:
        if n < 0:
            raise ValueError("n must be non-negative")
        x0 = self.x0_flat
        T = self.T_flat
        z = torch.randn(n, self.manifold_dim, device=x0.device, dtype=x0.dtype) * self.sample_std
        x = x0.unsqueeze(0) + z @ T
        return x.reshape(n, *self.ambient_shape)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        return self.distance(x)

    def get_normal_tangent_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_input_shape(x, name="x")
        batch_size = x.shape[0]
        N = self._N_on(x).unsqueeze(0).expand(batch_size, -1, -1)
        T = self._T_on(x).unsqueeze(0).expand(batch_size, -1, -1)
        N = N.reshape(batch_size, self.codim, *self.ambient_shape)
        T = T.reshape(batch_size, self.manifold_dim, *self.ambient_shape)
        return N, T

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x, name="x")
        self._validate_input_shape(v, name="v")
        v_flat = self._as_flat_batch(v)
        T = self._T_on(v_flat)
        coeff = v_flat @ T.transpose(-1, -2)
        v_tan = coeff @ T
        return v_tan.reshape(v.shape[0], *self.ambient_shape)

    def __repr__(self):
        return (
            f"AffineSubspace(dim={self.manifold_dim}, codim={self.codim}, "
            f"space={self.get_space()})"
        )


class SliceSubspace(AffineSubspace):
    """
    Selector-based affine subspace defined by fixed entries x[slice_spec] = b.

    This is a special case of AffineSubspace where A is a subset of rows from
    the identity matrix. Projection and tangent projection are implemented via
    direct indexed assignment/zeroing without dense linear algebra.
    """

    def __init__(
        self,
        slice_spec,
        b: torch.Tensor,
        ambient_shape: Tuple[int, ...] = None,
        space: Optional[Space] = None,
        sample_std: float = 1.0,
    ):
        if sample_std < 0:
            raise ValueError("sample_std must be non-negative")
        self.sample_std = float(sample_std)
        self.slice_spec = slice_spec

        if space is not None:
            if not isinstance(space, Space):
                raise TypeError(f"space must be a Space instance, got {type(space)}")
            space_shape = tuple(int(d) for d in space.get_dims())
            if ambient_shape is not None and tuple(int(d) for d in ambient_shape) != space_shape:
                raise ValueError(
                    f"ambient_shape {tuple(int(d) for d in ambient_shape)} "
                    f"does not match space dims {space_shape}"
                )
            ambient_shape_eff = space_shape
            Region.__init__(self, space)
        else:
            if ambient_shape is None:
                raise ValueError("ambient_shape must be provided when space is None")
            ambient_shape_eff = tuple(int(d) for d in ambient_shape)
            if len(ambient_shape_eff) == 0 or any(d <= 0 for d in ambient_shape_eff):
                raise ValueError(f"ambient_shape must be non-empty positive ints, got {ambient_shape}")
            ambient_space: Space = (
                VectorSpace(ambient_shape_eff[0])
                if len(ambient_shape_eff) == 1
                else EuclideanSpace(ambient_shape_eff)
            )
            Region.__init__(self, ambient_space)

        self.ambient_shape = tuple(ambient_shape_eff)
        self.ambient_dim = int(math.prod(self.ambient_shape))

        b_tensor = torch.as_tensor(b)
        if not b_tensor.is_floating_point():
            b_tensor = b_tensor.to(torch.float32)

        template = torch.arange(self.ambient_dim, device=b_tensor.device).reshape(self.ambient_shape)
        try:
            selected = template[slice_spec]
        except Exception as e:
            raise ValueError(f"Invalid slice_spec for ambient shape {self.ambient_shape}: {e}") from e

        selected_flat = selected.reshape(-1).to(dtype=torch.long)
        if selected_flat.numel() == 0:
            raise ValueError("slice_spec selects no entries")
        unique = torch.unique(selected_flat)
        if unique.numel() != selected_flat.numel():
            raise ValueError("slice_spec must select unique entries (no duplicates)")
        self.fixed_idx = selected_flat.contiguous()
        self.selection_shape = tuple(selected.shape)

        if b_tensor.numel() != self.fixed_idx.numel():
            raise ValueError(
                f"b has {b_tensor.numel()} entries but slice_spec selects {self.fixed_idx.numel()} entries"
            )
        if b_tensor.shape != self.selection_shape and b_tensor.ndim != 1:
            raise ValueError(
                f"b shape must match selected slice shape {self.selection_shape} or be flat, "
                f"got {tuple(b_tensor.shape)}"
            )
        self.b_flat = b_tensor.reshape(-1).contiguous()

        free_mask = torch.ones(self.ambient_dim, dtype=torch.bool, device=self.fixed_idx.device)
        free_mask[self.fixed_idx] = False
        self.free_idx = torch.arange(
            self.ambient_dim, dtype=torch.long, device=self.fixed_idx.device
        )[free_mask]

        self.codim = int(self.fixed_idx.numel())
        self.rank = self.codim
        self.manifold_dim = int(self.free_idx.numel())

        self.x0_flat = torch.zeros(
            self.ambient_dim, device=self.b_flat.device, dtype=self.b_flat.dtype
        )
        self.x0_flat[self.fixed_idx] = self.b_flat

    @classmethod
    def from_trajectory_space(
        cls,
        space: TrajectorySpace,
        slice_spec,
        b: torch.Tensor,
        sample_std: float = 1.0,
    ) -> "SliceSubspace":
        if not isinstance(space, TrajectorySpace):
            raise TypeError(f"space must be TrajectorySpace, got {type(space)}")
        return cls(
            slice_spec=slice_spec,
            b=b,
            space=space,
            sample_std=sample_std,
        )

    def _fixed_idx_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.fixed_idx.to(device=ref.device)

    def _free_idx_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.free_idx.to(device=ref.device)

    def _b_flat_on(self, ref: torch.Tensor) -> torch.Tensor:
        return self.b_flat.to(device=ref.device, dtype=ref.dtype)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x, name="x")
        x_flat = self._as_flat_batch(x).clone()
        x_flat[:, self._fixed_idx_on(x_flat)] = self._b_flat_on(x_flat).unsqueeze(0)
        return x_flat.reshape(x.shape[0], *self.ambient_shape)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x, name="x")
        self._validate_input_shape(v, name="v")
        v_flat = self._as_flat_batch(v).clone()
        v_flat[:, self._fixed_idx_on(v_flat)] = 0.0
        return v_flat.reshape(v.shape[0], *self.ambient_shape)

    def sample(self, n: int) -> torch.Tensor:
        if n < 0:
            raise ValueError("n must be non-negative")
        samples = torch.randn(n, *self.ambient_shape, dtype=self.b_flat.dtype) * self.sample_std
        flat = samples.reshape(n, self.ambient_dim)
        flat[:, self.fixed_idx] = self.b_flat.unsqueeze(0)
        return flat.reshape(n, *self.ambient_shape)

    def get_normal_tangent_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_input_shape(x, name="x")
        batch_size = x.shape[0]

        N_rows = torch.zeros(self.codim, self.ambient_dim, device=x.device, dtype=x.dtype)
        if self.codim > 0:
            N_rows[torch.arange(self.codim, device=x.device), self._fixed_idx_on(x)] = 1.0

        T_rows = torch.zeros(self.manifold_dim, self.ambient_dim, device=x.device, dtype=x.dtype)
        if self.manifold_dim > 0:
            T_rows[torch.arange(self.manifold_dim, device=x.device), self._free_idx_on(x)] = 1.0

        N = N_rows.unsqueeze(0).expand(batch_size, -1, -1)
        T = T_rows.unsqueeze(0).expand(batch_size, -1, -1)

        N = N.reshape(batch_size, self.codim, *self.ambient_shape)
        T = T.reshape(batch_size, self.manifold_dim, *self.ambient_shape)
        return N, T

    def __repr__(self):
        return (
            f"SliceSubspace(dim={self.manifold_dim}, codim={self.codim}, "
            f"selected={self.codim}, space={self.get_space()})"
        )


__all__ = ["Manifold", "AffineSubspace", "SliceSubspace"]
