/* Gaussian pixel filtering and the bounded perspective footprint. */
const GaussianFilter = (() => {
  const minimum = 1 / 65536;
  const cutoff = 4.5;
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

  function prepare(value) {
    if (value == null || value === false) return null;
    if (typeof value !== 'object' || Array.isArray(value) ||
        Object.keys(value).some(key => !['footprint', 'projection', 'variance'].includes(key)))
      throw new TypeError('Expected Gaussian footprint and projection settings');
    const { footprint = 'legacy', projection = 'affine', variance = .3 } = value;
    if (!['legacy', 'compensated'].includes(footprint) || !['affine', 'ray'].includes(projection) ||
        !Number.isFinite(variance) || variance < 0 || variance > 4)
      throw new RangeError('Gaussian filter needs legacy or compensated footprint, affine or ray projection, and variance in [0,4]');
    const compensated = footprint === 'compensated', ray = projection === 'ray';
    return compensated || ray ? { footprint, projection, variance, compensated, ray } : null;
  }

  function opacity(xx, xy, yy, variance) {
    // Use the determinant increment to avoid subtracting nearly equal products.
    const base = Math.max(0, xx*yy - xy*xy);
    const change = variance * (Math.max(0, xx + yy) + variance);
    return change > 0 ? Math.sqrt(base / (base + change)) : 1;
  }

  function cameraCovariance(cov, view) {
    const [xx, xy, xz, yy, yz, zz] = cov;
    const x = [view[0], view[4], view[8]], y = [view[1], view[5], view[9]], z = [-view[2], -view[6], -view[10]];
    const multiply = ([a,b,c]) => [xx*a + xy*b + xz*c, xy*a + yy*b + yz*c, xz*a + yz*b + zz*c];
    const u = multiply(x), v = multiply(y), w = multiply(z);
    return [dot(x,u), dot(y,u), dot(z,u), dot(y,v), dot(z,v), dot(z,w)];
  }

  function whitening([xx, xy, xz, yy, yz, zz]) {
    const floor = Math.max(xx, yy, zz, 1e-30) * 1e-7;
    const a = Math.sqrt(Math.max(xx, floor)), b = xy/a, c = xz/a;
    const d = Math.sqrt(Math.max(yy - b*b, floor)), e = (yz - b*c)/d;
    const f = Math.sqrt(Math.max(zz - c*c - e*e, floor));
    // Inverse of the lower triangular Cholesky factor. Floor the pivots
    // relative to the largest variance so flat splats remain evaluable.
    const factor = [1/a, -b/(a*d), 1/d, (b*e - c*d)/(a*d*f), -e/(d*f), 1/f];
    return { factor, covariance: [a*a, a*b, a*c, b*b+d*d, b*c+d*e, c*c+e*e+f*f] };
  }

  function limits(center, depth, variance, cross, depthVariance, focal) {
    const k = 2*cutoff;
    const a = depth*depth - k*depthVariance;
    const b = center*depth - k*cross;
    const discriminant = k*(depth*depth*variance - 2*depth*center*cross +
      center*center*depthVariance - k*(variance*depthVariance - cross*cross));
    const half = Math.sqrt(Math.max(0, discriminant));
    return [(b-half)*focal/a, (b+half)*focal/a];
  }

  function perspective(center, covariance, focal, near, far, variance) {
    const depth = center[2];
    // A tangent-plane blur adds the requested screen variance at the center.
    const blur = variance*(depth/focal)**2;
    const shape = covariance.slice();
    shape[0] += blur;
    shape[3] += blur;
    const { factor: m, covariance: c } = whitening(shape);
    const [x,y,z] = center;
    const p0 = m[0]*x, p1 = m[1]*x+m[2]*y, p2 = m[3]*x+m[4]*y+m[5]*z;
    const clipped = depth - 3*Math.sqrt(c[5]) <= near;
    const bounds = clipped ? null : [
      ...limits(x,depth,c[0],c[2],c[5],focal),
      ...limits(y,depth,c[3],c[4],c[5],focal)
    ];
    function power(offsetX, offsetY) {
      const rx = offsetX/focal, ry = offsetY/focal;
      const a = m[0]*rx, b = m[1]*rx+m[2]*ry, c = m[3]*rx+m[4]*ry+m[5];
      const t = Math.min(far, Math.max(near, (a*p0+b*p1+c*p2)/(a*a+b*b+c*c)));
      const dx = p0-a*t, dy = p1-b*t, dz = p2-c*t;
      return -.5*(dx*dx+dy*dy+dz*dz);
    }
    return { bounds, power };
  }

  const glsl = `
float gaussianOpacity(vec3 cov, float v) {
  float base = max(0.0, cov.x*cov.z - cov.y*cov.y);
  float change = v*(max(0.0,cov.x+cov.z)+v);
  return change > 0.0 ? sqrt(base/(base+change)) : 1.0;
}
vec2 gaussianLimits(float x, float z, float variance, float crossTerm, float zz, float focal) {
  float a = z*z - 9.0*zz, b = x*z - 9.0*crossTerm;
  float halfWidth = sqrt(max(0.0,9.0*(z*z*variance-2.0*z*x*crossTerm+x*x*zz-9.0*(variance*zz-crossTerm*crossTerm))));
  return (b+vec2(-halfWidth,halfWidth))*focal/a;
}
mat3 gaussianFactor(inout mat3 cov) {
  float eps = max(max(cov[0][0],cov[1][1]),max(cov[2][2],1e-30))*1e-7;
  float a = sqrt(max(cov[0][0],eps)), b = cov[0][1]/a, c = cov[0][2]/a;
  float d = sqrt(max(cov[1][1]-b*b,eps)), e = (cov[1][2]-b*c)/d;
  float f = sqrt(max(cov[2][2]-c*c-e*e,eps));
  mat3 L = mat3(a,b,c,0,d,e,0,0,f);
  cov = L*transpose(L);
  return mat3(1.0/a,-b/(a*d),(b*e-c*d)/(a*d*f),0,1.0/d,-e/(d*f),0,0,1.0/f);
}
`;

  return Object.freeze({ prepare, opacity, cameraCovariance, perspective, minimum, cutoff, glsl });
})();
if (typeof module !== 'undefined') module.exports = GaussianFilter;
else globalThis[Symbol.for('ink-studio.gaussian-filter')] = GaussianFilter;
