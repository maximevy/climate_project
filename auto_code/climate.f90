!----------------------------------------------------------------------
! Fortran 90 File for Single-Variable Climate Model
!----------------------------------------------------------------------
      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
! --------------------------------------------------
! Defines the right-hand side of the ODE: dT/dt = F(T, PAR)
!
! U(1) = T (Temperature)
! PAR(1) = ALPHA, PAR(2) = BETA, PAR(3) = LAMBDA, PAR(4) = C, PAR(5) = F
! --------------------------------------------------
      IMPLICIT NONE

      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
      DOUBLE PRECISION T, R

      DOUBLE PRECISION, PARAMETER :: PI = 3.1415926535897D0
      DOUBLE PRECISION, PARAMETER :: TPI = 2.D0 * 3.1415926535897D0
      
      ! Assign state variable
      T = U(1)

      ! Define the right-hand side (dT/dt)
      ! F(1) = (1/C) * (F + LAMBDA*T + ALPHA*T**2 + BETA*T**5)
      ! Using PAR indices directly:
      F(1) = (1/PAR(4)) * (PAR(5) + PAR(3)*T + PAR(1)*T**2 + PAR(2)*T**5)
      
      END SUBROUTINE FUNC


      SUBROUTINE STPNT(NDIM,U,PAR,T)
! --------------------------------------------------
! Defines the initial conditions and parameter values for the start of the run.
! --------------------------------------------------
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION, PARAMETER :: PI = 3.1415926535897D0

      ! Set Initial Parameters (Must use 1-based indexing)
      PAR(1) = 0.058D0          ! ALPHA
      PAR(2) = -4.0D-6       ! BETA (using D for double precision)
      PAR(3) = - 0.88D0         ! LAMBDA
      PAR(4) = 8.36D8         ! C
      PAR(5) = 0D0       ! F

      ! Set Initial Condition for State Variable U(1) = T
      U(1) = 0D0
      
      END SUBROUTINE STPNT


      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS(NDIM,U,PAR)
      ! --------------------------------------------------
      ! Subroutine for defining custom output measures (like Min/Max/Eigenvalues)
      ! Since this is not strictly necessary for a simple run, we can simplify this.
      ! --------------------------------------------------
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
      DOUBLE PRECISION, EXTERNAL :: GETP
      
      
      END SUBROUTINE PVLS