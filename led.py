import serial
import time

# CONFIGURATIE
PORT = 'COM6'          # Jouw poort
BAUD_RATE = 500000     # Standaard voor de meeste van deze strips
NUM_LEDS = 30          # Aantal LEDs om te testen (maakt voor test niet veel uit)

def calculate_checksum(count):
    # Adalight checksum formule: High byte XOR Low byte XOR 0x55
    high_byte = (count >> 8) & 0xFF
    low_byte = count & 0xFF
    return high_byte, low_byte, (high_byte ^ low_byte ^ 0x55)

try:
    print(f"Verbinden met {PORT}...")
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    # WACHTEN IS CRUCIAAL: Arduino-clones resetten vaak bij openen connectie
    time.sleep(3) 
    print("Verbinding open. Start testloop...")

    while True:
        # Loop door Rood, Groen, Blauw
        for r, g, b in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            
            # 1. Header opbouwen
            # Protocol: "Ada" + [Count Hi] + [Count Lo] + [Checksum]
            msg = bytearray([65, 100, 97]) # "Ada"
            
            count = NUM_LEDS - 1
            hi, lo, chk = calculate_checksum(count)
            msg.append(hi)
            msg.append(lo)
            msg.append(chk)
            
            # 2. Kleurdata toevoegen
            for _ in range(NUM_LEDS):
                msg.append(r)
                msg.append(g)
                msg.append(b)
                
            # 3. Verzenden
            ser.write(msg)
            print(f"Kleur gestuurd: {r}, {g}, {b}")
            time.sleep(1)

except serial.SerialException as e:
    print(f"FOUT: Kan poort niet openen. Is de andere software nog open? Sluit die eerst! \nError: {e}")
except KeyboardInterrupt:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print("\nTest gestopt.")