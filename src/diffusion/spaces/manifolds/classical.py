#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical manifold constraints (sphere, Stiefel, deformed torus)."""

import math
from typing import Tuple

import torch

from ..euclidean import MatrixSpace, VectorSpace
from .base import Manifold


class Sphere(Manifold):
    """
    n-dimensional sphere S^n ⊂ R^{n+1} ⊂ R^D.

    The sphere of dimension n is embedded in R^{n+1}, which may be
    embedded in a higher-dimensional ambient space R^D.

    The sphere is defined as:
        S^n = {x ∈ R^{n+1} : ||x|| = 1, x_{n+2} = ... = x_D = 0}

    Attributes:
        manifold_dim: Intrinsic dimension n
        ambient_dim: Full ambient dimension D
    """

    def __init__(self, manifold_dim: int, ambient_dim: int = None):
        """
        Initialize sphere.

        Args:
            manifold_dim: Intrinsic dimension n (dimension of sphere)
            ambient_dim: Ambient dimension D (if None, defaults to n+1)
        """
        if ambient_dim is None:
            ambient_dim = manifold_dim + 1

        assert ambient_dim > manifold_dim, "Ambient dimension must exceed manifold dimension"

        super().__init__(VectorSpace(ambient_dim))
        self.manifold_dim = manifold_dim
        self.ambient_dim = ambient_dim

    def get_dim_manifold(self) -> int:
        """Return intrinsic dimension n."""
        return self.manifold_dim

    def get_dim_ambient(self) -> int:
        """Return ambient dimension D."""
        return self.ambient_dim

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto the sphere.

        Projects first (n+1) coordinates onto unit sphere,
        sets remaining coordinates to zero.

        Args:
            x: (batch, D) points in ambient space

        Returns:
            x_proj: (batch, D) points on sphere
        """
        batch_size = x.shape[0]

        # Extract first (n+1) coordinates
        x_sphere = x[:, :self.manifold_dim + 1]

        # Normalize to unit sphere
        x_sphere = x_sphere / x_sphere.norm(dim=1, keepdim=True)

        # Pad with zeros
        x_proj = torch.cat(
            [
                x_sphere,
                torch.zeros(
                    batch_size,
                    self.ambient_dim - self.manifold_dim - 1,
                    device=x.device,
                ),
            ],
            dim=1,
        )

        return x_proj

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample uniformly from the sphere (Haar measure).

        Args:
            n: Number of samples

        Returns:
            samples: (n, D) points on sphere
        """
        # Sample from normal distribution on first (n+1) coords
        x_sphere = torch.randn(n, self.manifold_dim + 1)

        # Normalize
        x_sphere = x_sphere / x_sphere.norm(dim=1, keepdim=True)

        # Pad with zeros
        return torch.cat(
            [
                x_sphere,
                torch.zeros(n, self.ambient_dim - self.manifold_dim - 1),
            ],
            dim=1,
        )

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """Violation is Euclidean distance to sphere."""
        return self.distance(x)

    def get_normal_tangent_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get normal and tangent space bases at x.

        For sphere S^n ⊂ R^{n+1} ⊂ R^D:
        - Normal space: spanned by x (radial direction) and e_{n+2}, ..., e_D
        - Tangent space: orthogonal complement in first (n+1) coordinates

        Args:
            x: (batch, D) points on sphere

        Returns:
            N: (batch, D-n, D) - normal basis
            T: (batch, n, D) - tangent basis
        """
        # Project onto sphere first
        x = self.project(x)

        batch_size = x.shape[0]

        # Extract sphere coordinates
        x_sphere = x[:, :self.manifold_dim + 1]  # (batch, n+1)

        # Compute normal vectors in sphere coordinates
        # 1. Radial normal (normalized x)
        N_radial = x_sphere / x_sphere.norm(dim=1, keepdim=True)  # (batch, n+1)

        # 2. Standard basis for higher dimensions
        N_higher = torch.zeros(
            batch_size,
            self.ambient_dim - self.manifold_dim - 1,
            self.manifold_dim + 1,
            device=x.device,
        )  # (batch, D-n-1, n+1)

        # 3. Full normal basis in ambient space
        N_sphere = torch.cat([N_radial.unsqueeze(1), N_higher], dim=1)  # (batch, D-n, n+1)
        N_full = torch.zeros(
            batch_size,
            self.ambient_dim - self.manifold_dim,
            self.ambient_dim,
            device=x.device,
        )
        N_full[:, :, : self.manifold_dim + 1] = N_sphere
        N_full[:, 1:, self.manifold_dim + 1 :] = torch.eye(
            self.ambient_dim - self.manifold_dim - 1,
            device=x.device,
        ).unsqueeze(0)

        # Compute tangent basis via SVD
        # Tangent vectors in sphere coordinates
        xp = x_sphere  # (batch, n+1)
        Xp = xp.unsqueeze(2) @ xp.unsqueeze(1)  # (batch, n+1, n+1) outer product
        U, _, _ = torch.svd(Xp, some=False)
        T_sphere = U[:, :, 1:]  # (batch, n+1, n) - skip radial direction

        # Embed in full ambient space
        T_full = torch.zeros(batch_size, self.manifold_dim, self.ambient_dim, device=x.device)
        T_full[:, :, : self.manifold_dim + 1] = T_sphere.transpose(1, 2)

        return N_full, T_full

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto tangent space at x.

        For sphere, tangent space is orthogonal to radial direction.

        Args:
            x: (batch, D) points on sphere
            v: (batch, D) vectors in ambient space

        Returns:
            v_tangent: (batch, D) projected vectors
        """
        batch_size = x.shape[0]

        # Project x onto sphere
        x = self.project(x)

        # Extract sphere coordinates
        v_sphere = v[:, : self.manifold_dim + 1]
        x_sphere = x[:, : self.manifold_dim + 1]

        # Project onto tangent space: v - <v,x>x
        v_tangent_sphere = v_sphere - (v_sphere * x_sphere).sum(dim=1, keepdim=True) * x_sphere

        # Assemble full tangent vector
        v_tangent = torch.zeros(batch_size, self.ambient_dim, device=v.device)
        v_tangent[:, : self.manifold_dim + 1] = v_tangent_sphere

        return v_tangent

    def __repr__(self):
        return f"S^{self.manifold_dim} ⊂ R^{self.ambient_dim}"


class Stiefel(Manifold):
    """
    Stiefel manifold St(n,p) of n×p orthogonal matrices.

    The Stiefel manifold is:
        St(n,p) = {X ∈ R^{n×p} : X^T X = I_p}

    Attributes:
        n: Number of rows
        p: Number of columns
    """

    def __init__(self, n: int, p: int):
        """
        Initialize Stiefel manifold.

        Args:
            n: Number of rows
            p: Number of columns (p <= n)
        """
        assert n >= p, "Must have n >= p for Stiefel manifold"
        super().__init__(MatrixSpace(n, p))
        self.n = n
        self.p = p

    def get_dim_manifold(self) -> int:
        """Return intrinsic dimension: np - p(p+1)/2."""
        return self.n * self.p - self.p * (self.p + 1) // 2

    def get_dim_ambient(self) -> int:
        """Return ambient dimension: n*p."""
        return self.n * self.p

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project X onto Stiefel manifold via SVD.

        For X ∈ R^{n×p}, compute X = UΣV^T and return UV^T.

        Args:
            X: (batch, n, p) matrices

        Returns:
            X_proj: (batch, n, p) orthogonal matrices
        """
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return torch.bmm(U, Vh)

    def sample(self, n: int, uniform=True) -> torch.Tensor:
        """
        Sample from Stiefel manifold (not necessarily uniform).

        Uses QR decomposition of random matrices.

        Args:
            n: Number of samples

        Returns:
            samples: (n, self.n, self.p) orthogonal matrices
        """
        if uniform:
            assert self.n == self.p, "Not implemented for p != q"
            G = torch.randn(n, self.n, self.n)
            Q, R = torch.linalg.qr(G, mode="reduced")  # batch-capable
            d = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.sign(d)
            signs[signs == 0] = 1  # (probability zero under Gaussian, but be safe)
            D = torch.diag_embed(signs)
            return Q @ D
        X = torch.randn(n, self.n, self.p)
        U, _, _ = torch.linalg.svd(X, full_matrices=False)
        return U

    def violation(self, X: torch.Tensor) -> torch.Tensor:
        """Violation is Frobenius distance to Stiefel manifold."""
        return self.distance(X)

    def get_normal_tangent_space(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get normal and tangent space bases at X.

        Tangent space: T_X St(n,p) = {X*Ω + X_⊥*K : Ω skew-symmetric}
        Normal space: N_X St(n,p) = {X*S : S symmetric}

        Args:
            X: (batch, n, p) points on Stiefel

        Returns:
            N: (batch, k_normal, n, p) - normal basis
            T: (batch, k_tangent, n, p) - tangent basis
        """
        # Project onto Stiefel first
        X = self.project(X)
        batch_size = X.shape[0]

        # Compute X_perp via QR
        Q, _ = torch.linalg.qr(X, mode="complete")  # (batch, n, n)
        X_perp = Q[:, :, self.p :]  # (batch, n, n-p)

        # Pre-compute basis generators (no batch dim)
        S_sym = self._sym_basis()  # (k, p, p)
        A_skew = self._skew_basis()  # (q, p, p)
        B_vert = self._vertical_basis()  # (r, n-p, p)

        k = S_sym.shape[0]
        q = A_skew.shape[0]
        r = B_vert.shape[0]

        # Normal basis: N[b,k] = X[b] @ S_sym[k]
        N = torch.einsum("bnp,kpq->bknq", X, S_sym)  # (batch, k, n, p)

        # Tangent basis (skew part)
        T_skew = torch.einsum("bnp,qpr->bqnr", X, A_skew)  # (batch, q, n, p)

        # Tangent basis (vertical part)
        if r == 0:  # n == p case
            T = T_skew
        else:
            T_vert = torch.einsum("bns,rsp->brnp", X_perp, B_vert)  # (batch, r, n, p)
            T = torch.cat([T_skew, T_vert], dim=1)  # (batch, q+r, n, p)

        return N, T

    def _sym_basis(self) -> torch.Tensor:
        """
        Frobenius-orthonormal symmetric generators.

        Returns:
            (k, p, p) where k = p(p+1)/2
        """
        mats = []
        sqrt2_inv = 1.0 / math.sqrt(2.0)

        for i in range(self.p):
            S = torch.zeros((self.p, self.p))
            S[i, i] = 1.0
            mats.append(S)

            for j in range(i + 1, self.p):
                S = torch.zeros((self.p, self.p))
                S[i, j] = S[j, i] = sqrt2_inv
                mats.append(S)

        return torch.stack(mats)

    def _skew_basis(self) -> torch.Tensor:
        """
        Frobenius-orthonormal skew-symmetric generators.

        Returns:
            (q, p, p) where q = p(p-1)/2
        """
        mats = []
        sqrt2_inv = 1.0 / math.sqrt(2.0)

        for i in range(self.p):
            for j in range(i + 1, self.p):
                A = torch.zeros((self.p, self.p))
                A[i, j] = sqrt2_inv
                A[j, i] = -sqrt2_inv
                mats.append(A)

        return torch.stack(mats) if mats else torch.empty((0, self.p, self.p))

    def _vertical_basis(self) -> torch.Tensor:
        """
        Canonical vertical generators.

        Returns:
            (r, n-p, p) where r = (n-p)*p
        """
        if self.n == self.p:  # No vertical component
            return torch.empty((0, 0, self.p))

        eye_left = torch.eye(self.n - self.p)
        eye_right = torch.eye(self.p)

        verts = [
            torch.outer(eye_left[l], eye_right[j])
            for l in range(self.n - self.p)
            for j in range(self.p)
        ]

        return torch.stack(verts)  # (r, n-p, p)

    def project_tangent(self, X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Project G onto tangent space at X.

        Uses formula: P_X(G) = (I - XX^T)G + X*skew(X^T G)

        Args:
            X: (batch, n, p) points on Stiefel
            G: (batch, n, p) matrices in ambient space

        Returns:
            G_tangent: (batch, n, p) tangent vectors
        """
        PX = torch.eye(self.n, device=X.device).unsqueeze(0) - torch.bmm(X, X.transpose(-1, -2))
        S = torch.bmm(X.transpose(-1, -2), G)
        S = (S - S.transpose(-1, -2)) / 2  # Skew-symmetrize
        return torch.bmm(PX, G) + torch.bmm(X, S)

    def __repr__(self):
        return f"St({self.n},{self.p})"


class DeformedTorus(Manifold):
    """
    Deformed torus M = Psi(T_{R,r}) in R^3, where

        Psi(x,y,z) = (x, y + alpha sin(x), z + beta sin(2x))

    and T_{R,r} is the standard torus with major radius R and minor radius r.

    Shapes:
        points x: (..., 3)
        vectors v: (..., 3)

    Notes
    -----
    - project_tangent is exact.
    - project is a pullback-based projection:
          x -> Psi( Proj_torus( Psi^{-1}(x) ) )
      This is exact for the standard torus in the pulled-back coordinates, but
      not the exact Euclidean closest-point projection on the deformed torus
      unless Psi is an isometry.
    """

    def __init__(
        self,
        R: float = 1.2,
        r: float = 0.45,
        alpha: float = 0.35,
        beta: float = 0.20,
        eps: float = 1e-8,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(VectorSpace(3))
        if R <= 0:
            raise ValueError("R must be positive.")
        if r <= 0:
            raise ValueError("r must be positive.")
        if r >= R:
            raise ValueError("Require r < R for a standard ring torus.")

        self.R = float(R)
        self.r = float(r)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype

    def get_dim_manifold(self) -> int:
        return 2

    def get_dim_ambient(self) -> int:
        return 3

    def _psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3)
        """
        x0 = x[..., 0]
        x1 = x[..., 1] + self.alpha * torch.sin(x0)
        x2 = x[..., 2] + self.beta * torch.sin(2.0 * x0)
        return torch.stack([x0, x1, x2], dim=-1)

    def _psi_inv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Explicit inverse of Psi:
            (X,Y,Z) -> (X, Y - alpha sin(X), Z - beta sin(2X))
        """
        x0 = x[..., 0]
        x1 = x[..., 1] - self.alpha * torch.sin(x0)
        x2 = x[..., 2] - self.beta * torch.sin(2.0 * x0)
        return torch.stack([x0, x1, x2], dim=-1)

    def _J_psi_inv_T(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return J_{Psi^{-1}}(x)^T as a batch of matrices with shape (..., 3, 3).
        """
        X = x[..., 0]
        c1 = -self.alpha * torch.cos(X)
        c2 = -2.0 * self.beta * torch.cos(2.0 * X)

        shape = x.shape[:-1] + (3, 3)
        Jt = torch.zeros(shape, dtype=x.dtype, device=x.device)

        # transpose of J_inv
        Jt[..., 0, 0] = 1.0
        Jt[..., 0, 1] = c1
        Jt[..., 0, 2] = c2
        Jt[..., 1, 1] = 1.0
        Jt[..., 2, 2] = 1.0
        return Jt

    def _torus_implicit(self, y: torch.Tensor) -> torch.Tensor:
        """
        Implicit function for the standard torus:
            F(x,y,z) = (x^2 + y^2 + z^2 + R^2 - r^2)^2 - 4 R^2 (x^2 + y^2)
        """
        x0, x1, x2 = y[..., 0], y[..., 1], y[..., 2]
        s = x0**2 + x1**2 + x2**2 + self.R**2 - self.r**2
        return s**2 - 4.0 * self.R**2 * (x0**2 + x1**2)

    def _grad_torus_implicit(self, y: torch.Tensor) -> torch.Tensor:
        """
        Gradient of the standard torus implicit function.
        """
        x0, x1, x2 = y[..., 0], y[..., 1], y[..., 2]
        s = x0**2 + x1**2 + x2**2 + self.R**2 - self.r**2

        gx = 4.0 * x0 * (s - 2.0 * self.R**2)
        gy = 4.0 * x1 * (s - 2.0 * self.R**2)
        gz = 4.0 * x2 * s
        return torch.stack([gx, gy, gz], dim=-1)

    def _project_to_standard_torus(self, y: torch.Tensor) -> torch.Tensor:
        """
        Closed-form geometric projection onto the standard torus.
        """
        x0, x1, x2 = y[..., 0], y[..., 1], y[..., 2]
        rho = torch.sqrt(x0**2 + x1**2 + self.eps)

        # Closest point on the central circle
        c0 = self.R * x0 / rho
        c1 = self.R * x1 / rho
        c2 = torch.zeros_like(x2)
        c = torch.stack([c0, c1, c2], dim=-1)

        w = y - c
        nw = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(self.eps)

        return c + self.r * w / nw

    def _normal_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normal to the deformed torus at x.
        """
        y = self._psi_inv(x)
        gradF = self._grad_torus_implicit(y)  # (..., 3)
        Jt = self._J_psi_inv_T(x)  # (..., 3, 3)
        n = torch.matmul(Jt, gradF.unsqueeze(-1)).squeeze(-1)  # (..., 3)
        return n

    def _unit_normal(self, x: torch.Tensor) -> torch.Tensor:
        n = self._normal_unnormalized(x)
        n_norm = torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(self.eps)
        return n / n_norm

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Orthogonal projection of v onto T_x M:
            v_tan = v - <v,n> n
        where n is the unit normal of the deformed torus at x.
        """
        if x.shape[-1] != 3 or v.shape[-1] != 3:
            raise ValueError("x and v must have last dimension 3.")
        n = self._unit_normal(x)
        vn = (v * n).sum(dim=-1, keepdim=True)
        return v - vn * n

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pull back to the standard torus, project there, then map back.
        """
        if x.shape[-1] != 3:
            raise ValueError("x must have last dimension 3.")

        orig_shape = x.shape
        x_flat = x.reshape(-1, 3)

        y = self._psi_inv(x_flat)
        y_proj = self._project_to_standard_torus(y)
        return self._psi(y_proj).reshape(orig_shape)

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points on the deformed torus by sampling the standard torus
        parameters (u,v) and then applying Psi.

        This samples parameters uniformly, not surface area exactly uniformly.
        """
        if n <= 0:
            raise ValueError("n must be positive.")

        device = self.device
        dtype = self.dtype

        u = 2.0 * math.pi * torch.rand(n, device=device, dtype=dtype)
        v = 2.0 * math.pi * torch.rand(n, device=device, dtype=dtype)

        cu, su = torch.cos(u), torch.sin(u)
        cv, sv = torch.cos(v), torch.sin(v)

        x0 = (self.R + self.r * cv) * cu
        x1 = (self.R + self.r * cv) * su
        x2 = self.r * sv

        y = torch.stack([x0, x1, x2], dim=-1)
        return self._psi(y)

    def violation(self, x):
        return self.distance(x)

    def get_normal_tangent_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get orthonormal bases for normal and tangent spaces at x.

        Args:
            x: (..., 3) points on manifold

        Returns:
            N: (..., 1, 3) normal basis vectors
            T: (..., 2, 3) tangent basis vectors
        """
        if x.shape[-1] != 3:
            raise ValueError("x must have last dimension 3.")

        # Unit normal basis
        n = self._unit_normal(x)  # (..., 3)
        N = n.unsqueeze(-2)  # (..., 1, 3)

        # Build two orthonormal tangent vectors by choosing a reference axis
        # that is not too aligned with the normal.
        e1 = torch.zeros_like(n)
        e1[..., 0] = 1.0

        e2 = torch.zeros_like(n)
        e2[..., 1] = 1.0

        # Use e1 unless n is too close to e1, then use e2
        use_e2 = torch.abs(n[..., 0]) > 0.9
        a = torch.where(use_e2.unsqueeze(-1), e2, e1)  # (..., 3)

        # First tangent vector
        t1 = a - (a * n).sum(dim=-1, keepdim=True) * n
        t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True).clamp_min(self.eps)

        # Second tangent vector
        t2 = torch.cross(n, t1, dim=-1)
        t2 = t2 / torch.linalg.norm(t2, dim=-1, keepdim=True).clamp_min(self.eps)

        T = torch.stack([t1, t2], dim=-2)  # (..., 2, 3)

        return N, T


__all__ = ["Sphere", "Stiefel", "DeformedTorus"]
