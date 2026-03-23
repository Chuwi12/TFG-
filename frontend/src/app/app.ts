import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly title = signal('Financial Termianl');

  // serhio voy a crear un señal con la lista d emensajaes incial
  protected readonly messages = signal([
    { text: 'SYSTEM READY. Authorized access only.', sender: 'system' },
    { text: 'Waiting for financial query...', sender: 'system' },
  ]);

  // funciona para aadirm los mensajes del usuario
  addMessage(input: HTMLInputElement) {
    const value = input.value.trim();
    if (value) {
      // actualizar señal
      this.messages.update((prev) => [...prev, { text: value, sender: 'user' }]);
      input.value = '';
    }
  }
}
