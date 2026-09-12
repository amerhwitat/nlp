#!/usr/bin/env perl
use strict; use warnings; use File::Basename qw(dirname); use Cwd qw(abs_path); use File::Spec;
my $root=abs_path(File::Spec->catdir(dirname(__FILE__),'..','..')); chdir $root or die "Cannot enter $root: $!";
my $mode=shift(@ARGV)||'check';
if($mode eq 'check'){ system($^O eq 'MSWin32'?'where':'command','sqlite3')==0 || print "sqlite3 not found\n"; system($^O eq 'MSWin32'?'where':'command','psql')==0 || print "psql not found\n"; exit 0; }
die "Usage: init-database.pl check|init\n" unless $mode eq 'init';
mkdir 'artifacts' unless -d 'artifacts'; mkdir 'artifacts/database' unless -d 'artifacts/database';
my $db='artifacts/database/nlp.sqlite';
if(system('sqlite3','--version')==0){ for my $f (glob('db/sql/*.sql')){ system('sqlite3',$db,".read $f")==0 or die "Failed $f\n"; } } else { die "Install sqlite3 or configure PostgreSQL through the Python/PowerShell initializer.\n"; }
print "Database initialization completed.\n";
