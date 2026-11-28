#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/module_symmetry/symmetry.h"
#include "source_pw/module_dfpt/dfpt_symm.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

namespace
{

/// fcc primitive lattice with cubic side a = 1 (lat0 = 1 units), the
/// diamond-structure cell of the production Si cases
ModuleBase::Matrix3 fcc_latvec() { return ModuleBase::Matrix3(0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0); }

/// the 48 signed-permutation cartesian rotations (the Oh point group)
std::vector<ModuleBase::Matrix3> cubic_cartesian_ops()
{
    std::vector<ModuleBase::Matrix3> ops;
    const int rows[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};
    for (int p = 0; p < 6; ++p)
    {
        for (int sx = 1; sx >= -1; sx -= 2)
        {
            for (int sy = 1; sy >= -1; sy -= 2)
            {
                for (int sz = 1; sz >= -1; sz -= 2)
                {
                    const int signs[3] = {sx, sy, sz};
                    double m[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    for (int i = 0; i < 3; ++i)
                    {
                        m[i * 3 + rows[p][i]] = signs[i];
                    }
                    ops.push_back(ModuleBase::Matrix3(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]));
                }
            }
        }
    }
    return ops;
}

/// the Oh rotations expressed in the fcc primitive fractional basis
/// (integer matrices, the gmatrix convention of the symmetry module)
std::vector<ModuleBase::Matrix3> fcc_gmatrices()
{
    const ModuleBase::Matrix3 lat = fcc_latvec();
    const ModuleBase::Matrix3 inv_lat = lat.Inverse();
    const std::vector<ModuleBase::Matrix3> ops = cubic_cartesian_ops();
    std::vector<ModuleBase::Matrix3> gm;
    for (const ModuleBase::Matrix3 &r : ops)
    {
        gm.push_back(lat * r * inv_lat);
    }
    return gm;
}

} // namespace

TEST(DFPTLittleGroup, GammaKeepsAllOperations)
{
    const std::vector<ModuleBase::Matrix3> gm = fcc_gmatrices();
    const ModuleBase::Vector3<double> q(0.0, 0.0, 0.0);
    const std::vector<int> group = ModuleDFPT::little_group_of_q(fcc_latvec(), gm.data(), gm.size(), q);
    EXPECT_EQ(group.size(), gm.size());
}

TEST(DFPTLittleGroup, LPointLittleGroupIsD3dOfTwelve)
{
    // L = (pi/a)(-1, 1, 1) with a = 1: the little group including time
    // reversal is D3d (12 operations); the dot-product reciprocal test
    // must reject C2y, whose cartesian difference (1, 0, -1)*2pi/a is
    // integral but not a reciprocal-lattice vector
    const std::vector<ModuleBase::Matrix3> gm = fcc_gmatrices();
    const ModuleBase::Vector3<double> q(-M_PI, M_PI, M_PI);
    const std::vector<int> group = ModuleDFPT::little_group_of_q(fcc_latvec(), gm.data(), gm.size(), q);
    EXPECT_EQ(group.size(), 12u);
    // the identity operation is always a member
    bool has_identity = false;
    for (const int iop : group)
    {
        if (std::abs(gm[iop].e11 - 1.0) < 1.0e-12 && std::abs(gm[iop].e22 - 1.0) < 1.0e-12 &&
            std::abs(gm[iop].e33 - 1.0) < 1.0e-12)
        {
            has_identity = true;
        }
    }
    EXPECT_TRUE(has_identity);
}

TEST(DFPTLittleGroup, XPointLittleGroupIsD4hOfSixteen)
{
    // X = (0, 2pi/a, 0) with a = 1: the little group including time
    // reversal is D4h (16 operations: the maps with the y column +-y)
    const std::vector<ModuleBase::Matrix3> gm = fcc_gmatrices();
    const ModuleBase::Vector3<double> q(0.0, 2.0 * M_PI, 0.0);
    const std::vector<int> group = ModuleDFPT::little_group_of_q(fcc_latvec(), gm.data(), gm.size(), q);
    EXPECT_EQ(group.size(), 16u);
}

TEST(DFPTLittleGroup, GenericQKeepsIdentityAndInversionTimesTR)
{
    // for a generic q the unitary branch keeps only the identity, but the
    // antiunitary branch always admits inversion combined with time
    // reversal ((theta*i) q = q for every q), so the gray little group of
    // any q has at least two members
    const std::vector<ModuleBase::Matrix3> gm = fcc_gmatrices();
    const ModuleBase::Vector3<double> q(1.234, -0.567, 0.891);
    const std::vector<int> group = ModuleDFPT::little_group_of_q(fcc_latvec(), gm.data(), gm.size(), q);
    EXPECT_EQ(group.size(), 2u);
    EXPECT_EQ(group.front(), 0);
}

TEST(DFPTLittleGroup, EmptyInputIsEmptyOutput)
{
    const ModuleBase::Vector3<double> q(0.0, 0.0, 0.0);
    const std::vector<int> group = ModuleDFPT::little_group_of_q(fcc_latvec(), nullptr, 0, q);
    EXPECT_TRUE(group.empty());
}

namespace
{

/// a minimal serial rho basis on the cubic test cell (the kq_basis test
/// pattern, trimmed to what rhog_rotate_one needs: the G-list, the
/// stick/FFT maps and the grid dims)
struct RhoGrid
{
    ModulePW::PW_Basis pw;
    std::vector<int> ixyz2ipw;
    double lat0 = 1.8897261254578281;
    double cubic_edge = 10.0;
};

RhoGrid build_rho_grid()
{
    RhoGrid rg;
    const double b = ModuleBase::TWO_PI / rg.lat0;
    rg.pw.nx = 7;
    rg.pw.ny = 7;
    rg.pw.nz = 7;
    rg.pw.nxyz = 343;
    rg.pw.fftny = 7;
    rg.pw.fftnx = 7;
    rg.pw.fftnz = 7;
    rg.pw.fftnxyz = 343;
    rg.pw.gamma_only = false;
    rg.pw.G = ModuleBase::Matrix3(b, 0, 0, 0, b, 0, 0, 0, b);
    rg.pw.ggecut = 16.0 * b * b / 4.0; // the 4*ecutwfc ball, ~2 shells
    // FillSticks pattern: enumerate the box, keep the (ix, iy) columns
    // intersecting the ball, fill the z runs
    std::vector<int> is2fftixy;
    std::vector<int> ig2isz;
    auto wrap = [](int i, int n) { return (i > n / 2) ? i - n : i; };
    for (int ix0 = 0; ix0 < 7; ++ix0)
    {
        for (int iy0 = 0; iy0 < 7; ++iy0)
        {
            const int wix = wrap(ix0, 7);
            const int wiy = wrap(iy0, 7);
            bool has = false;
            for (int iz0 = 0; iz0 < 7; ++iz0)
            {
                const int wiz = wrap(iz0, 7);
                const double g2 = (wix * wix + wiy * wiy + wiz * wiz) * b * b;
                if (g2 <= rg.pw.ggecut)
                {
                    has = true;
                    break;
                }
            }
            if (!has)
            {
                continue;
            }
            const int is = static_cast<int>(is2fftixy.size());
            is2fftixy.push_back(ix0 * 7 + iy0);
            for (int iz0 = 0; iz0 < 7; ++iz0)
            {
                const int wiz = wrap(iz0, 7);
                const double g2 = (wix * wix + wiy * wiy + wiz * wiz) * b * b;
                if (g2 <= rg.pw.ggecut)
                {
                    ig2isz.push_back(is * 7 + iz0);
                }
            }
        }
    }
    rg.pw.nst = static_cast<int>(is2fftixy.size());
    rg.pw.npw = static_cast<int>(ig2isz.size());
    rg.pw.is2fftixy = new int[rg.pw.nst];
    rg.pw.ig2isz = new int[rg.pw.npw];
    for (int i = 0; i < rg.pw.nst; ++i)
    {
        rg.pw.is2fftixy[i] = is2fftixy[i];
    }
    for (int i = 0; i < rg.pw.npw; ++i)
    {
        rg.pw.ig2isz[i] = ig2isz[i];
    }
    // the grid -> G-list map of the rhog_symmetry callers
    rg.ixyz2ipw.assign(rg.pw.fftnxyz, -1);
    for (int ig = 0; ig < rg.pw.npw; ++ig)
    {
        const int isz = rg.pw.ig2isz[ig];
        const int iz = isz % 7;
        const int is = isz / 7;
        const int ixy = rg.pw.is2fftixy[is];
        const int ix = ixy / 7;
        const int iy = ixy % 7;
        rg.ixyz2ipw[(ix * rg.pw.fftny + iy) * rg.pw.fftnz + iz] = ig;
    }
    return rg;
}

} // namespace

TEST(DFPTRhogRotate, GaussianDensityMapsToTheRotatedCenter)
{
    // the convention oracle: a density f(r) = exp(-|r - r0|^2 / 2 s^2)
    // has G-space coefficients c(n) ~ exp(-2 pi i n . r0_frac) rad(n) with
    // rad rotation-invariant, so the {R|t} image of f is the Gaussian at
    // {R|t} r0 = R r0 + t and the rotated coefficients must satisfy
    // c'(m) = c(G) exp(-2 pi i m . t) with m = kg * G (integer grid
    // rotation). The cubic ops are orthogonal so m . (R r0) = G . r0 and
    // the radial factor cancels.
    RhoGrid rg = build_rho_grid();
    const double b = ModuleBase::TWO_PI / rg.lat0;
    const double edge = rg.cubic_edge;
    // a few off-center Gaussian positions (fractional)
    const std::vector<ModuleBase::Vector3<double>> centers = {ModuleBase::Vector3<double>(0.13, 0.21, 0.34),
        ModuleBase::Vector3<double>(0.05, 0.27, 0.11), ModuleBase::Vector3<double>(0.22, 0.08, 0.41)};
    // ops: C4z, the xy mirror, inversion, and a C2 with a fractional
    // translation (any t is a valid probe for the rotation primitive)
    const std::vector<ModuleBase::Matrix3> ops = {ModuleBase::Matrix3(0, -1, 0, 1, 0, 0, 0, 0, 1),
        ModuleBase::Matrix3(1, 0, 0, 0, 1, 0, 0, 0, -1), ModuleBase::Matrix3(-1, 0, 0, 0, -1, 0, 0, 0, -1),
        ModuleBase::Matrix3(0, 1, 0, 1, 0, 0, 0, 0, -1)};
    const std::vector<ModuleBase::Vector3<double>> trans = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0),
        ModuleBase::Vector3<double>(0.0, 0.0, 0.0), ModuleBase::Vector3<double>(0.0, 0.0, 0.0),
        ModuleBase::Vector3<double>(0.25, 0.0, 0.125)};
    const double sigma_frac2 = 0.02; // narrow but grid-resolvable
    for (size_t ic = 0; ic < centers.size(); ++ic)
    {
        for (size_t iop = 0; iop < ops.size(); ++iop)
        {
            const ModuleBase::Matrix3 &r = ops[iop];
            const ModuleBase::Vector3<double> &t = trans[iop];
            // expected center in fractional coordinates: r0' = r r0 + t
            const ModuleBase::Vector3<double> r0 = centers[ic];
            const ModuleBase::Vector3<double> expect_center(
                r.e11 * r0.x + r.e12 * r0.y + r.e13 * r0.z + t.x, r.e21 * r0.x + r.e22 * r0.y + r.e23 * r0.z + t.y,
                r.e31 * r0.x + r.e32 * r0.y + r.e33 * r0.z + t.z);
            std::vector<std::complex<double>> c_in(rg.pw.npw, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> c_out(rg.pw.npw, std::complex<double>(0.0, 0.0));
            // c(n) = exp(-2 pi i n . r0) exp(-sigma |n|^2) on the G-list
            for (int ixyz = 0; ixyz < rg.pw.fftnxyz; ++ixyz)
            {
                const int ipw = rg.ixyz2ipw[ixyz];
                if (ipw < 0)
                {
                    continue;
                }
                const int k = ixyz % rg.pw.fftnz;
                const int j = ((ixyz - k) / rg.pw.fftnz) % rg.pw.fftny;
                const int i = ((ixyz - k) / rg.pw.fftnz - j) / rg.pw.fftny;
                const int gx = (i > rg.pw.nx / 2 + 1) ? (i - rg.pw.nx) : i;
                const int gy = (j > rg.pw.ny / 2 + 1) ? (j - rg.pw.ny) : j;
                const int gz = (k > rg.pw.nz / 2 + 1) ? (k - rg.pw.nz) : k;
                const double arg = -ModuleBase::TWO_PI * (gx * r0.x + gy * r0.y + gz * r0.z);
                const double radial = std::exp(-sigma_frac2 * (gx * gx + gy * gy + gz * gz));
                c_in[ipw] = std::complex<double>(std::cos(arg), std::sin(arg)) * radial;
            }
            // the module stores the reciprocal rep kg = R^T for the
            // direct-space op R, and the primitive consumes that
            // convention; for the orthogonal cubic ops R^T = R^-1
            const ModuleBase::Matrix3 kg = ops[iop].Transpose();
            ModuleSymmetry::Symmetry::rhog_rotate_one(
                c_in.data(), c_out.data(), rg.ixyz2ipw.data(), rg.pw.nx, rg.pw.ny, rg.pw.nz, rg.pw.fftnx, rg.pw.fftny,
                rg.pw.fftnz, kg, t);
            (void)edge;
            // verify: c_out(m) = exp(-2 pi i m . expect_center) radial(|m|),
            // scanning the grid points that carry coefficients
            for (int ixyz = 0; ixyz < rg.pw.fftnxyz; ++ixyz)
            {
                const int ipw = rg.ixyz2ipw[ixyz];
                if (ipw < 0)
                {
                    continue;
                }
                const int k = ixyz % rg.pw.fftnz;
                const int j = ((ixyz - k) / rg.pw.fftnz) % rg.pw.fftny;
                const int i = ((ixyz - k) / rg.pw.fftnz - j) / rg.pw.fftny;
                const int gx = (i > rg.pw.nx / 2 + 1) ? (i - rg.pw.nx) : i;
                const int gy = (j > rg.pw.ny / 2 + 1) ? (j - rg.pw.ny) : j;
                const int gz = (k > rg.pw.nz / 2 + 1) ? (k - rg.pw.nz) : k;
                const double arg =
                    -ModuleBase::TWO_PI * (gx * expect_center.x + gy * expect_center.y + gz * expect_center.z);
                const double radial = std::exp(-sigma_frac2 * (gx * gx + gy * gy + gz * gz));
                const std::complex<double> expect = std::complex<double>(std::cos(arg), std::sin(arg)) * radial;
                const double scale = std::max(1.0, std::abs(expect));
                EXPECT_NEAR(c_out[ipw].real(), expect.real(), 1.0e-9 * scale);
                EXPECT_NEAR(c_out[ipw].imag(), expect.imag(), 1.0e-9 * scale);
            }
        }
    }
}
