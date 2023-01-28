import pygame
import neat
import time
import os
import random
pygame.font.init()
# Window has width 600 and height 800
WIND_WIDTH = 575
WIND_HEIGHT = 800

GEN = 0

# Importing images
BIRD_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "b1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "b2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "b3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    # Bird behaviour constants
    IMG = BIRD_IMG
    M_ROTATION = 25
    ROT_SPEED = 20
    ANIMATION_t = 3

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.speed = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMG[0]
    
    # Bird flaps!
    def flap(self):
        self.speed = -10.5
        self.tick_count = 0
        self.hight = self.y

    # Bird moves when it flaps (and when it doesn't)
    def move(self):
        self.tick_count += 1
        disp_x = self.speed * self.tick_count + 1.5*self.tick_count**2

        # Making sure bird stays between bounds
        if disp_x >= 16:
            disp_x = 16
        if disp_x < 0:
            disp_x -= 2
        
        self.y += disp_x

        if disp_x < 0 or self.y < self.height + 50:
            if self.tilt < self.M_ROTATION:
                self.tilt = self.M_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_SPEED

    # Show accurate movement when the bird is flying
    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_t:
            self.img = self.IMG[0]
        elif self.img_count < self.ANIMATION_t*2:
            self.img = self.IMG[1]
        elif self.img_count < self.ANIMATION_t*3:
            self.img = self.IMG[2]
        elif self.img_count < self.ANIMATION_t*4:
            self.img = self.IMG[1]
        elif self.img_count == self.ANIMATION_t*4 + 1:
            self.img = self.IMG[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMG[1]
            self.img_count = self.ANIMATION_t*2

        # Bird rotates around center. Got from stackoverflow but can't find the link :(
        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_img, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    # Pipe behaviour constants
    PGAP = 200
    SPD = 5
    # Initialize pipes!
    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    # Position of the pipes in the window
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.PGAP

    # Pipe moves toward the left
    def move(self):
        self.x -= self.SPD

    # Show the pipe!
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # Check if bird hits pipe
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False

class Base:
    SPD = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.SPD
        self.x2 -= self.SPD

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 +self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

# Actually draw bird, pipes, and base in the window
def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

        text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
        win.blit(text, (WIND_WIDTH - 10 - text.get_width(), 10))

        text = STAT_FONT.render("Gen: " + str(gen), 1, (255,255,255))
        win.blit(text, (10, 10))
    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def Neural_Eval(genomes, config):
    global GEN
    GEN += 1
    networks = []
    genome = []
    birds = []

    # Characterize each neural network to a bird
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        genome.append(g)

    base = Base(730)
    pipes = [Pipe(575)]
    win = pygame.display.set_mode((WIND_WIDTH, WIND_HEIGHT))
    clock = pygame.time.Clock()

    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False 
                pygame.quit()
                quit()


        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        # Reward bird for going forward
        for x, bird in enumerate(birds):
            bird.move()
            genome[x].fitness += 0.1

            output = networks[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.flap()

        add_pipe = False
        removed_pipes = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                # Punish birds that hit the pipe
                if pipe.collide(bird):
                    genome[x].fitness -= 1
                    birds.pop(x)
                    networks.pop(x)
                    genome.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed_pipes.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            # Reward birds that mke it through pipe
            for g in genome:
                g.fitness += 1
            pipes.append(Pipe(575))
        for r in removed_pipes:
            pipes.remove(r)

        # Punish birds that try to cheat the system by flying over or under the map
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                networks.pop(x)
                genome.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score, GEN)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(Neural_Eval, 50)
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)