import '@angular/compiler';
import { describe, it, expect } from 'vitest';
import { App } from './app';

describe('App', () => {
  it('should define the root component', () => {
    expect(App).toBeTruthy();
  });

  it('should expose the expected chat action', () => {
    expect(App.prototype.sendMessage).toBeTypeOf('function');
  });
});
