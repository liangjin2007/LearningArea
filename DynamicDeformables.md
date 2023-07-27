## Deformation function and Deformation energy function


- A deformation can be represented by φ(x) : x -> x,     R3 to R3.

- F: deformation gradient -- The derivative is d φ(x) / dx, this is a higher level tensor F.
```
∂φ(x¯)/∂x¯
=∂(Fx¯ + t)/∂x¯
= F.
```

- Score function out of F Ψ(F) : when F = I it should be zero. It's a scalar function. Or it's energy function
  - Frobenius norm ||F||, also known as Dirichlet energy. wrong
  - Ψ Neo-Hookean = ||F||^2 - 3       wrong
  - ΨNeo-Hookean,Band-Aid =    (||F||^2 - 3)^2        wrong
  - ΨC&SL = ||F − I||^2                               wrong
  - ΨStVK, stretch = || F^T F - I ||^2                good
    - = ||FT F||^2 +tr I - 2 tr(FT F)                 ?? why
- C = FT F The right Cauchy-Green tensor
- E = 1/2(C - I) Green’s strain
  
- polar decomposition F = R S, not very safe
```
Usually R is only defined as a unitary matrix, not a rotation matrix, so it only needs to satisfy RT R = I.
This means that there could be a reflection lurking somewhere in R, whereas we want R to be a pure rotation.
In our case, if a reflection has to lurk somewhere, we would prefer that it do so in S. We will call this specific flavor the rotation variant of the polar decomposition
```
- ΨARAP = ||F-R||^2, ARAP
  - = ||F||^2 + 3 - 2tr(S)
- corotational energy
  - stiffness warping
  - rotated linear 

## Calculate force
- f = -area dΨ(F)/dx, this is derivative related to x, so need to know dF/dx
- Computing ∂Ψ/∂x
- F = Ds Dm ^ -1， It's quite similar with the LBS bindings, Dm^-1 can be taken as transformation to local space.

## Calculate Force the Tensor Way

## Calculate the Force gradient the Tensor Way

## A Better way for Isotropic Solids

## A Friendlier Neo-Hookean Energy

## The analytical EigenSystem of Isotropic Energy

## A Better way of Anisotropic Solids

## Tips for Computing and Debugging Force Derivatives

## Thin Shell Forces

## Implicit Integration Method

## Constrained Backward Euler

## Collision Processing

## Collision Energy


