import * as shared from '../gaussian-filter.js';

export const gaussianFilter = shared.default ?? globalThis[Symbol.for('ink-studio.gaussian-filter')];
