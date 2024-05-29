import numpy as np 
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import pygame
def get_energy(lattice):
    convolution_mask = generate_binary_structure(2,1)
    convolution_mask[1,1] = False
    Energy =  -lattice*convolve(lattice,convolution_mask, mode = "wrap")/2
    Energy = Energy.sum()
    return Energy

def get_lattice(L,spinRatio = 0.5):
    random = np.random.random((L,L))
    lattice = np.zeros((L,L))
    lattice[random>=spinRatio] = 1
    lattice[random<spinRatio] = -1
    return lattice

@njit
def fastsweepgame(lattice ,beta, startEnergy):
    energy = startEnergy
    for t in range(1,len(lattice)**2):
        
        E_p = 0
        E_t = 0
        old_lattice = lattice.copy()
        i = np.random.randint(0,len(lattice)) 
        j = np.random.randint(0,len(lattice))
        flipped_spin = lattice[i,j] 
        lattice[i,j] = flipped_spin*(-1)
        for n in range(2):
            E_t += lattice[i,(j+(-1)**n)%len(lattice)]
            E_t += lattice[(i+(-1)**n)%len(lattice),j]
          

        E_n = -E_t*lattice[i,j]
        E_p = -E_t*flipped_spin
        deltaE = E_n-E_p
        if np.exp(-beta*deltaE) > np.random.random():
            energy = deltaE + energy
        else:
            lattice = old_lattice
            
            
      
    return lattice, energy
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.critrect = pygame.Rect(x+w*0.44069,y,3,h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.font  = pygame.font.SysFont("Times New Roman", 18)
        
         

    def draw(self, screen):
       
        pygame.draw.rect(screen, (200,200,200), self.rect)
        pygame.draw.rect(screen,(255,0,0),self.critrect)
        handle_pos = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w
        pygame.draw.circle(screen, (0,255,0), (int(handle_pos), self.rect.centery), self.rect.h // 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Update the slider value
                rel_x = event.pos[0] - self.rect.x
                self.value = self.min_val + (rel_x / self.rect.w) * (self.max_val - self.min_val)
                self.value = max(self.min_val, min(self.value, self.max_val))
    def show_beta(self,screen,x,y):
        beta = self.value
        betarect = pygame.Rect(x,y,60,20)
        pygame.draw.rect(screen,(200,200,200),betarect)
        beta_gui = self.font.render(str(round(beta,3)),1,(255,0,0))
        screen.blit(beta_gui,(x+5,y))
def main():
    beta = 0.7
    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Ising Model')
    lattice = get_lattice(100)
    energy = get_energy(lattice)
    rows, cols = lattice.shape[0], lattice.shape[1]  
    cell_width = width // cols
    cell_height = height // rows


    norm_lattice = np.zeros((len(lattice),len(lattice)))
    norm_lattice[lattice<0] = 0
    # Color map function (grayscale)
    def value_to_color(value):
        gray = int(value * 255)
        return (gray, gray, gray)
    slider = Slider(100,700,600,40,0.01,1,0.5)
    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            slider.handle_event(event)
        beta = slider.value
        screen.fill((0, 0, 0))
        for i in range(rows):
            for j in range(cols):
                color = value_to_color(norm_lattice[i][j])
                pygame.draw.rect(
                    screen,
                    color,
                    [j * cell_width, i * cell_height, cell_width, cell_height]
                )
        slider.draw(screen)
        slider.show_beta(screen,370,750)

        pygame.display.flip()
        clock.tick(30)
        lattice , energy = fastsweepgame(lattice,beta,energy)
        norm_lattice = (lattice - np.min(lattice)) / (np.max(lattice) - np.min(lattice))

            # Quit Pygame
    pygame.quit()
        
if __name__ == '__main__':
    main()