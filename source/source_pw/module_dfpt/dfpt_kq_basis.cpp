#include "dfpt_kq_basis.h"

#include "source_base/global_function.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"

#include <set>

namespace ModuleDFPT
{

DFPT_KQ_Basis::DFPT_KQ_Basis()
{
}

DFPT_KQ_Basis::~DFPT_KQ_Basis()
{
}

void DFPT_KQ_Basis::init(const ModulePW::PW_Basis_K* pw_wfc,
                         const ModulePW::PW_Basis* pw_rho,
                         const ModuleBase::Vector3<double>& q_cart,
                         int ik)
{
    ModuleBase::TITLE("DFPT_KQ_Basis", "init");
    ModuleBase::timer::start("DFPT_KQ_Basis", "init");
    pw_wfc_ = pw_wfc;
    npwk_ = 0;
    ig_rho_.clear();
    gk2_.clear();
    gcar_.clear();

    if (pw_wfc_ == nullptr || pw_rho == nullptr)
    {
        ModuleBase::timer::end("DFPT_KQ_Basis", "init");
        return;
    }

    // DFPT couples k and k+q symmetrically; the perturbation wavevector is
    // generally incommensurate with the gamma-ladder, so the ground-state
    // basis must be a full complex basis (see class documentation).
    if (pw_wfc_->gamma_only)
    {
        ModuleBase::WARNING_QUIT("DFPT_KQ_Basis",
                                 "DFPT requires a complex (gamma_only=false) wavefunction basis for k+q. "
                                 "Please disable gamma_only for the wavefunction basis used by DFPT.");
    }

    // the two bases exchange G vectors through the shared FFT cell position
    if (pw_wfc_->nx != pw_rho->nx || pw_wfc_->ny != pw_rho->ny || pw_wfc_->nz != pw_rho->nz)
    {
        ModuleBase::WARNING_QUIT("DFPT_KQ_Basis",
                                 "DFPT requires the wavefunction and charge FFT grids to share "
                                 "their dimensions.");
    }

    // commensurate fast path: when k+q matches a ground-state k' on the
    // mesh (k' = k + q + dn for an integer reciprocal-basis triple dn),
    // the k+q ball is exactly the k' ball translated — same Cartesian
    // momenta, same count. This guarantees N(k+q) == N(k') and avoids
    // the two-reservoir boundary mismatch.
    const ModuleBase::Matrix3 ginv = pw_wfc_->G.Inverse();
    const ModuleBase::Vector3<double> q_frac = q_cart * ginv;
    const ModuleBase::Vector3<double> target = pw_wfc_->kvec_d[ik] + q_frac;
    for (int j = 0; j < pw_wfc_->nks; ++j)
    {
        const ModuleBase::Vector3<double>& kj = pw_wfc_->kvec_d[j];
        const double dx = kj.x - target.x;
        const double dy = kj.y - target.y;
        const double dz = kj.z - target.z;
        const double rx = std::round(dx);
        const double ry = std::round(dy);
        const double rz = std::round(dz);
        constexpr double mtol = 1.0e-6;
        if (std::abs(dx - rx) < mtol && std::abs(dy - ry) < mtol && std::abs(dz - rz) < mtol)
        {
            const ModuleBase::Vector3<int> dn(static_cast<int>(rx),
                                               static_cast<int>(ry),
                                               static_cast<int>(rz));
            init_from_gs_ball(pw_wfc, pw_rho, q_cart, ik, j, dn);
            ModuleBase::timer::end("DFPT_KQ_Basis", "init");
            return;
        }
    }
    // no commensurate match: fall through to the two-reservoir enumeration

    const ModuleBase::Vector3<double> k_c = pw_wfc_->kvec_c[ik];
    kplusq_c_ = k_c + q_cart;

    // rho-grid reverse map of the shared FFT cell, used to attach the
    // charge-basis index of every enumerated k+q plane wave
    std::vector<int> ig_of_cell(pw_rho->nxyz, -1);
    for (int ig = 0; ig < pw_rho->npw; ++ig)
    {
        const int isz = pw_rho->ig2isz[ig];
        const int iz = isz % pw_rho->nz;
        const int is = isz / pw_rho->nz;
        const int ixy = pw_rho->is2fftixy[is];
        const int ix = ixy / pw_rho->fftny;
        const int iy = ixy % pw_rho->fftny;
        ig_of_cell[(ix * pw_rho->ny + iy) * pw_rho->nz + iz] = ig;
    }

    std::set<int> taken;
    auto try_push = [&](const int ix_in,
                        const int iy_in,
                        const int iz_in,
                        const ModuleBase::Matrix3& gbase,
                        const int ig_rho_hint) {
        int ix = ix_in;
        int iy = iy_in;
        int iz = iz_in;
        if (ix >= static_cast<int>(pw_wfc_->nx / 2) + 1)
        {
            ix -= pw_wfc_->nx;
        }
        if (iy >= static_cast<int>(pw_wfc_->ny / 2) + 1)
        {
            iy -= pw_wfc_->ny;
        }
        if (iz >= static_cast<int>(pw_wfc_->nz / 2) + 1)
        {
            iz -= pw_wfc_->nz;
        }
        const ModuleBase::Vector3<double> gcar = ModuleBase::Vector3<double>(ix, iy, iz) * gbase;
        const ModuleBase::Vector3<double> gpluskq = gcar + kplusq_c_;
        const double gk2 = gpluskq * gpluskq;
        if (gk2 > pw_wfc_->gk_ecut)
        {
            return;
        }
        const int cell = ((ix_in * pw_wfc_->ny) + iy_in) * pw_wfc_->nz + iz_in;
        if (!taken.insert(cell).second)
        {
            return;
        }
        gcar_.push_back(gcar);
        gk2_.push_back(gk2);
        ig_rho_.push_back(ig_rho_hint >= 0 ? ig_rho_hint : ig_of_cell[cell]);
    };

    // first reservoir: the wavefunction G grid (covers the ground-state
    // k-mesh balls)
    for (int ig = 0; ig < pw_wfc_->npw; ++ig)
    {
        const int isz = pw_wfc_->ig2isz[ig];
        const int iz = isz % pw_wfc_->nz;
        const int is = isz / pw_wfc_->nz;
        const int ixy = pw_wfc_->is2fftixy[is];
        const int ix = ixy / pw_wfc_->fftny;
        const int iy = ixy % pw_wfc_->fftny;
        try_push(ix, iy, iz, pw_wfc_->G, -1);
    }
    // completing reservoir: the charge G grid; its ball radius
    // (2*sqrt(ecutwfc) by construction) covers every q-shifted ball for q
    // inside the first Brillouin zone, which the wavefunction grid does not
    for (int ig = 0; ig < pw_rho->npw; ++ig)
    {
        const int isz = pw_rho->ig2isz[ig];
        const int iz = isz % pw_rho->nz;
        const int is = isz / pw_rho->nz;
        const int ixy = pw_rho->is2fftixy[is];
        const int ix = ixy / pw_rho->fftny;
        const int iy = ixy % pw_rho->fftny;
        try_push(ix, iy, iz, pw_rho->G, ig);
    }
    npwk_ = static_cast<int>(gcar_.size());
    ModuleBase::timer::end("DFPT_KQ_Basis", "init");
}

void DFPT_KQ_Basis::clear()
{
    ModuleBase::TITLE("DFPT_KQ_Basis", "clear");
    ModuleBase::timer::start("DFPT_KQ_Basis", "clear");
    pw_wfc_ = nullptr;
    kplusq_c_ = ModuleBase::Vector3<double>();
    npwk_ = 0;
    ig_rho_.clear();
    gk2_.clear();
    gcar_.clear();
    ModuleBase::timer::end("DFPT_KQ_Basis", "clear");
}


void DFPT_KQ_Basis::init_from_gs_ball(const ModulePW::PW_Basis_K* pw_wfc,
                                       const ModulePW::PW_Basis* pw_rho,
                                       const ModuleBase::Vector3<double>& q_cart,
                                       int ik,
                                       int ikq,
                                       const ModuleBase::Vector3<int>& dn)
{
    ModuleBase::TITLE("DFPT_KQ_Basis", "init_from_gs_ball");
    ModuleBase::timer::start("DFPT_KQ_Basis", "init_from_gs_ball");
    pw_wfc_ = pw_wfc;
    npwk_ = 0;
    ig_rho_.clear();
    gk2_.clear();
    gcar_.clear();

    if (pw_wfc_ == nullptr || pw_rho == nullptr)
    {
        ModuleBase::timer::end("DFPT_KQ_Basis", "init_from_gs_ball");
        return;
    }

    if (pw_wfc_->gamma_only)
    {
        ModuleBase::WARNING_QUIT("DFPT_KQ_Basis",
                                 "DFPT requires a complex (gamma_only=false) wavefunction basis for k+q.");
    }

    kplusq_c_ = pw_wfc_->kvec_c[ik] + q_cart;

    // dn in Cartesian: the integer reciprocal-basis triple times the
    // reciprocal lattice matrix gives the Cartesian G_recip of the fold
    const ModuleBase::Vector3<double> dn_cart = 
        ModuleBase::Vector3<double>(dn.x, dn.y, dn.z) * pw_wfc_->G;

    // rho-grid reverse map (same construction as in init())
    std::vector<int> ig_of_cell(pw_rho->nxyz, -1);
    for (int ig = 0; ig < pw_rho->npw; ++ig)
    {
        const int isz = pw_rho->ig2isz[ig];
        const int iz = isz % pw_rho->nz;
        const int is = isz / pw_rho->nz;
        const int ixy = pw_rho->is2fftixy[is];
        const int ix = ixy / pw_rho->fftny;
        const int iy = ixy % pw_rho->fftny;
        ig_of_cell[(ix * pw_rho->ny + iy) * pw_rho->nz + iz] = ig;
    }

    // The k+q ball is exactly the k' ball translated by dn_cart: same
    // Cartesian momenta (|k+q+G|^2 == |k'+G'|^2), same plane-wave count.
    // No cutoff re-check needed — every G' in the k' ball already
    // satisfies |G'+k'|^2 < gk_ecut by construction.
    const int nkq = pw_wfc_->npwk[ikq];
    npwk_ = nkq;
    gcar_.resize(nkq);
    gk2_.resize(nkq);
    ig_rho_.resize(nkq, -1);

    const ModuleBase::Matrix3 ginv = pw_wfc_->G.Inverse();

    for (int jgl = 0; jgl < nkq; ++jgl)
    {
        // G' from the ground-state k' ball (Cartesian)
        const ModuleBase::Vector3<double> gprime = pw_wfc_->getgcar(ikq, jgl);

        // G = G' + dn_cart (translate to the k+q ball)
        gcar_[jgl] = gprime + dn_cart;

        // |G + (k+q)|^2 == |G' + k'|^2 (same Cartesian momentum)
        const ModuleBase::Vector3<double> gpluskq = gcar_[jgl] + kplusq_c_;
        gk2_[jgl] = gpluskq * gpluskq;

        // Map to the charge-grid FFT cell for the ig_rho linkage:
        // convert G to fractional reciprocal-basis integers, wrap to the
        // FFT grid range [0, n), and look up the charge grid reverse map
        const ModuleBase::Vector3<double> gf = gcar_[jgl] * ginv;
        int ix = static_cast<int>(std::round(gf.x));
        int iy = static_cast<int>(std::round(gf.y));
        int iz = static_cast<int>(std::round(gf.z));
        ix = ((ix % pw_rho->nx) + pw_rho->nx) % pw_rho->nx;
        iy = ((iy % pw_rho->ny) + pw_rho->ny) % pw_rho->ny;
        iz = ((iz % pw_rho->nz) + pw_rho->nz) % pw_rho->nz;
        const int cell = (ix * pw_rho->ny + iy) * pw_rho->nz + iz;
        if (cell >= 0 && cell < static_cast<int>(ig_of_cell.size()))
        {
            ig_rho_[jgl] = ig_of_cell[cell];
        }
    }

    ModuleBase::timer::end("DFPT_KQ_Basis", "init_from_gs_ball");
}

} // namespace ModuleDFPT
